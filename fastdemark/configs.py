from pathlib import Path



ROOT = Path(__file__).parent.parent

RESOURCES_DIR = ROOT / "resources"
RESOURCES_DIR.mkdir(exist_ok=True)




if __name__ == "__main__":
    from loguru import logger
    logger.debug(ROOT)