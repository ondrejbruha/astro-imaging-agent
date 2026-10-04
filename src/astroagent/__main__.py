"""Enable python -m astroagent as an alternative to the aia entry point."""

from astroagent.cli.main import app

if __name__ == "__main__":
    app()
