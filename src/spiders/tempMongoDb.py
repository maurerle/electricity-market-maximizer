from pymongo import MongoClient
import pprint

# TODO add pymongo[srv] to requirements
MONGO = 'mongodb+srv://new-user:politomerda@cluster0-awyti.mongodb.net/test?retryWrites=true&w=majority'

# Mongo Client Init
client = MongoClient(
    MONGO
)
db = client['InterProj']
print(db.list_collection_names())
pprint.pprint(db['MGPFabbisogno'])