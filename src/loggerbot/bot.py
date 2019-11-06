import sys
import telepot
from src.common.config import *

sys.dont_write_bytecode = True

def bot(level, obj, msg): 
    logger = telepot.Bot(TOKEN)
    for id in CHAT_IDS:
        try:
            logger.sendMessage(id, 
                f"[{level}] [{obj}] {msg}",
            )
        except:
            pass