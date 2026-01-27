"""
Project entrypoint that runs the API app in seal_embedding_api.main
"""

import uvicorn


def main():
    uvicorn.run(
        "seal_embedding_api.main:app",
        host="0.0.0.0",
        port=8000,
        reload=False,
        log_level="info",
    )


if __name__ == "__main__":
    main()
