import os
from datetime import datetime, timezone
import logging

logger = logging.getLogger("mongo_db")

def get_mongo_client():
    mongo_uri = os.environ.get("MONGO_URI", "mongodb://localhost:27017")
    try:
        from pymongo import MongoClient
        client = MongoClient(mongo_uri, serverSelectionTimeoutMS=2000)
        # Test connection
        client.admin.command('ping')
        return client
    except Exception as e:
        logger.warning(f"MongoDB connection failed: {e}")
        return None

def get_db():
    client = get_mongo_client()
    if client:
        db_name = os.environ.get("MONGO_DB_NAME", "hospital_assistant")
        return client[db_name]
    return None

def get_database_status():
    client = get_mongo_client()
    if client:
        try:
            db_name = os.environ.get("MONGO_DB_NAME", "hospital_assistant")
            db = client[db_name]
            count = db.chat_logs.count_documents({})
            return True, f"Connected to MongoDB ({db_name}) | Total Logs: {count}"
        except Exception as e:
            return False, f"MongoDB Error: {e}"
    return False, "MongoDB Disconnected (Operating in In-Memory Mode)"

def save_chat_log(session_id: str, user_query: str, intent: str, retrieved_context: str, response: str):
    db = get_db()
    if db is not None:
        try:
            log_entry = {
                "session_id": session_id,
                "user_query": user_query,
                "intent": intent,
                "retrieved_context": retrieved_context,
                "final_response": response,
                "timestamp": datetime.now(timezone.utc)
            }
            db.chat_logs.insert_one(log_entry)
            return True
        except Exception as e:
            logger.error(f"Failed to insert chat log into MongoDB: {e}")
            return False
    return False

def get_recent_logs(limit: int = 20):
    db = get_db()
    if db is not None:
        try:
            logs = list(db.chat_logs.find({}, {"_id": 0}).sort("timestamp", -1).limit(limit))
            return logs
        except Exception as e:
            logger.error(f"Failed to fetch logs from MongoDB: {e}")
            return []
    return []

def get_intent_analytics():
    db = get_db()
    if db is not None:
        try:
            pipeline = [
                {"$group": {"_id": "$intent", "count": {"$sum": 1}}},
                {"$sort": {"count": -1}}
            ]
            results = list(db.chat_logs.aggregate(pipeline))
            return {item["_id"]: item["count"] for item in results}
        except Exception as e:
            logger.error(f"Failed to calculate analytics: {e}")
            return {}
    return {}
