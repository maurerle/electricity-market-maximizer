from sys import dont_write_bytecode
from telepot import Bot
from src.common.config import TOKEN, CHAT_IDS

dont_write_bytecode = True

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
    logger = Bot(TOKEN)
    for id in CHAT_IDS:
        try:
            logger.sendMessage(id, 
                f"[{level}] [{obj}] {msg}",
            )
        except:
            pass