import sys
import telepot
from common.config import *

sys.dont_write_bytecode = True

def bot(level, obj, msg): 
    logger = telepot.Bot(TOKEN)
    for id in CHAT_IDS:
        logger.sendMessage(523755114, 
			f"[{level}] [{obj}] {msg}",
		)