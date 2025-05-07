import os
import tempfile
import logging
import httpx  # Asynchronous HTTP client
from fastapi import FastAPI, HTTPException, Depends, Request, status, BackgroundTasks
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from pydantic import BaseModel, HttpUrl, Field
from pdfminer.high_level import extract_text
from google.oauth2 import id_token
import google.auth
from google.auth.transport.requests import AuthorizedSession, Request as GoogleAuthRequest # Renamed to avoid conflict
import cachecontrol  # Improves performance of google-auth
from dotenv import load_dotenv  # For local .env file loading

# --- Load environment variables from .env file (for local development with docker-compose) ---
# Docker will typically pass environment variables directly, but this doesn't hurt.
load_dotenv()

# --- Configuration ---
# EXPECTED_AUDIENCE will be loaded from environment variables (set in .env for local,
# or via Docker environment variables / systemd service file on VM)
EXPECTED_AUDIENCE = os.environ.get("EXPECTED_AUDIENCE")

# Configure logging
logging.basicConfig(
    level=logging.INFO, # Consider logging.DEBUG for more verbosity if needed
    format="%(asctime)s - %(name)s - %(levelname)s - %(process)d - %(threadName)s - %(message)s",
)
logger = logging.getLogger(__name__)

if not EXPECTED_AUDIENCE:
    logger.warning(
        "EXPECTED_AUDIENCE environment variable not set. "
        "Authentication will fail unless this is for unauthenticated testing or "
        "the API is configured to bypass auth for certain paths."
    )

# --- Pydantic Models ---
class PdfUrlRequest(BaseModel):
    pdfUrl: HttpUrl = Field(..., examples=["https://www.w3.org/WAI/ER/tests/xhtml/testfiles/resources/pdf/dummy.pdf"])

class TextResponse(BaseModel):
    text: str

class ErrorDetail(BaseModel):
    error_type: str
    message: str

class ErrorResponse(BaseModel):
    detail: ErrorDetail

# --- Authentication ---
# Use CacheControl with a standard requests session for fetching Google's public keys
# as Application Default Credentials (ADC) might not be available or relevant
# just for token verification.
try:
    http_session = httpx.Client() # Standard httpx client for fetching keys
    cached_http_session = cachecontrol.CacheControl(http_session)
    google_token_request_adapter = GoogleAuthRequest(session=cached_http_session)
    logger.info("Configured google.auth.transport.requests.Request with CacheControl for ID token verification.")
except Exception as e:
    logger.error(f"Failed to setup cached session for google-auth: {e}", exc_info=True)
    # Fallback to a basic request adapter if CacheControl setup fails
    google_token_request_adapter = GoogleAuthRequest()


bearer_scheme = HTTPBearer(
    description="Google ID Token obtained from Apps Script's `ScriptApp.getIdentityToken()`"
)

async def verify_google_id_token(
    auth: HTTPAuthorizationCredentials = Depends(bearer_scheme)
) -> str:
    """
    Verifies Google ID token and returns the authenticated identity (e.g., email or sub).
    """
    token = auth.credentials
    if not EXPECTED_AUDIENCE:
        logger.error("Authentication attempt failed: Server configuration error (EXPECTED_AUDIENCE missing)")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail={"error_type": "config_error", "message": "Server authentication configuration error."},
        )
    try:
        # For debugging, NEVER log the full token in production if sensitive.
        # logger.debug(f"Attempting to validate token: {token[:20]}...") # Log first 20 chars

        id_info = id_token.verify_oauth2_token(
            token, google_token_request_adapter, EXPECTED_AUDIENCE
        )
        # 'sub' is a stable identifier for the user. 'email' might also be present.
        identity = id_info.get('sub') or id_info.get('email')
        if not identity:
            logger.error(f"Identity (sub/email) not found in validated token. Claims: {id_info}")
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail={"error_type": "invalid_token", "message": "Identity not found in token."}
            )
        logger.info(f"Request authenticated for identity: {identity}")
        return identity
    except ValueError as e:
        logger.error(f"Token validation failed: {e}")
        # logger.debug(f"Failed token for context: {token}") # Debugging: log failed token
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail={"error_type": "invalid_token", "message": str(e)},
            headers={"WWW-Authenticate": "Bearer error=\"invalid_token\""},
        )
    except Exception as e:
        logger.error(f"An unexpected error occurred during token verification: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail={"error_type": "auth_error", "message": "An internal error occurred during authentication."},
        )

# --- FastAPI App Instance ---
app = FastAPI(
    title="PDF Text Extraction API",
    description="Extracts text from PDF files provided via URL. Requires Google ID Token authentication.",
    version="0.1.0",
    # Add root_path if running behind a reverse proxy with a path prefix
    # root_path="/api/v1" # Example if Traefik routes /api/v1 to this app
)

# --- Helper function to delete temp file in background ---
def cleanup_temp_file(path: str):
    try:
        if path and os.path.exists(path):
            os.remove(path)
            logger.info(f"Successfully deleted temporary file: {path}")
    except OSError as e:
        logger.error(f"Error deleting temporary file {path}: {e}")

# --- API Endpoints ---
@app.get("/health", summary="Health check endpoint", status_code=status.HTTP_200_OK)
async def health_check():
    """
    Simple health check endpoint.
    """
    logger.info("Health check endpoint called.")
    return {"status": "healthy"}

@app.post(
    "/extract_pdf",
    response_model=TextResponse,
    summary="Extract text from a PDF URL",
    responses={
        status.HTTP_400_BAD_REQUEST: {"model": ErrorResponse, "description": "Invalid request or PDF download/processing failure"},
        status.HTTP_401_UNAUTHORIZED: {"description": "Invalid or missing authentication token"},
        status.HTTP_500_INTERNAL_SERVER_ERROR: {"model": ErrorResponse, "description": "Internal server error"},
    },
)
async def extract_pdf_from_url_endpoint(
    payload: PdfUrlRequest,
    background_tasks: BackgroundTasks,
    # Enforce authentication for this endpoint
    authenticated_identity: str = Depends(verify_google_id_token)
):
    """
    Downloads a PDF from the provided URL, extracts its text content,
    and returns the extracted text.
    Authentication via Google ID Token is required.
    """
    pdf_url = str(payload.pdfUrl) # Convert Pydantic HttpUrl to string for httpx
    temp_pdf_path = None
    logger.info(f"Authenticated identity '{authenticated_identity}' initiated PDF extraction for URL: {pdf_url}")

    try:
        # 1. Download the PDF asynchronously with streaming
        # Using httpx.AsyncClient for async requests
        async with httpx.AsyncClient(timeout=60.0) as client: # Timeout for download
            logger.info(f"Attempting to download PDF from: {pdf_url}")
            try:
                async with client.stream("GET", pdf_url, follow_redirects=True) as response:
                    response.raise_for_status() # Raise HTTPStatusError for 4xx/5xx responses

                    # 2. Save to temporary file
                    # Using tempfile.NamedTemporaryFile for secure temporary file creation
                    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf", dir="/tmp") as temp_file_obj:
                        temp_pdf_path = temp_file_obj.name
                        downloaded_bytes = 0
                        async for chunk in response.aiter_bytes(chunk_size=8192):
                            temp_file_obj.write(chunk)
                            downloaded_bytes += len(chunk)
                        logger.info(f"PDF ({downloaded_bytes} bytes) downloaded temporarily to: {temp_pdf_path}")

            except httpx.HTTPStatusError as e:
                logger.error(f"HTTP error downloading PDF from {pdf_url}: {e.response.status_code} - {e.response.text}")
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail={"error_type": "download_failed", "message": f"Could not fetch PDF: HTTP {e.response.status_code}"}
                )
            except httpx.RequestError as e: # Catches network errors, timeouts, etc.
                logger.error(f"Network error downloading PDF from {pdf_url}: {e}")
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail={"error_type": "download_failed", "message": f"Could not fetch PDF: Network error - {type(e).__name__}"}
                )

        # 3. Extract text from the temporary PDF file
        try:
            logger.info(f"Attempting to extract text from temporary file: {temp_pdf_path}")
            # pdfminer.six's extract_text is synchronous, consider running in a thread pool for CPU-bound tasks
            # if performance becomes an issue with many concurrent requests.
            # For now, direct call is fine.
            extracted_text = extract_text(temp_pdf_path)
            logger.info(f"Successfully extracted text (length: {len(extracted_text)}) from: {pdf_url}")
            return {"text": extracted_text}
        except Exception as e: # Catch specific pdfminer exceptions if known, otherwise general
            logger.error(f"Error during PDF text extraction from {temp_pdf_path}: {e}", exc_info=True)
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, # Or 400 if it's a bad PDF
                detail={"error_type": "extraction_failed", "message": f"Failed to extract text from PDF: {type(e).__name__}"}
            )
        finally:
            # 4. Schedule background task to delete the temporary file
            if temp_pdf_path: # Ensure path exists before scheduling deletion
                background_tasks.add_task(cleanup_temp_file, temp_pdf_path)
                logger.info(f"Background task scheduled to delete: {temp_pdf_path}")

    except HTTPException:
        # Re-raise HTTPExceptions that were already handled (e.g., from download)
        raise
    except Exception as e:
        logger.error(f"An unexpected error occurred in extract_pdf_from_url_endpoint: {e}", exc_info=True)
        # Ensure temp_pdf_path is cleaned up even on unexpected errors if it was created
        if temp_pdf_path:
             background_tasks.add_task(cleanup_temp_file, temp_pdf_path)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail={"error_type": "unexpected_error", "message": f"An unexpected server error occurred."}
        )

# --- For local development: uvicorn main:app --reload ---
if __name__ == "__main__":
    import uvicorn
    logger.info("Starting Uvicorn server for local development...")
    # Note: --reload is typically passed via CLI, not here programmatically for Gunicorn.
    # Gunicorn will manage workers.
    uvicorn.run(app, host="0.0.0.0", port=8000)
