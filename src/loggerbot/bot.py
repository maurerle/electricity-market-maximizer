import sys
import telepot
from src.common.config import *

sys.dont_write_bytecode = True

def bot(level, obj, msg): 
    """Telegram bot obtain WARNING level logs remotely.
    
    Parameters
    ----------
    level : str
        logging level
    obj : str
        code's section (e.g. for the spider obj = GME or Terna)
    msg : str
        error message
    """
    logger = telepot.Bot(TOKEN)
    for id in CHAT_IDS:
        try:
            logger.sendMessage(id, 
                f"[{level}] [{obj}] {msg}",
            )
        except:
            pass