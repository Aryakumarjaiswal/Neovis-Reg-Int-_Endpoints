
import json
import pymysql
from datetime import datetime  


def create_database_and_table(data):
    conn = pymysql.connect(
        host="localhost",
        user="root",  
        password="#1Krishna",  
        database="Conversations"  
    )
    cursor = conn.cursor()

    
    cursor.execute(
        """
        CREATE TABLE IF NOT EXISTS bookings_info (
            _id VARCHAR(255),
            platform VARCHAR(255),
            platform_id VARCHAR(255),
            listing_id VARCHAR(255),
            confirmation_code VARCHAR(255),
            check_in DATETIME,
            check_out DATETIME,
            listing_title VARCHAR(255),
            account_id VARCHAR(255),
            guest_id VARCHAR(255),
            guest_name VARCHAR(255),
            commission DECIMAL(10, 2)
        )
        """
    )

    
    for entry in data:
        
        check_in = datetime.strptime(entry["checkIn"], "%Y-%m-%dT%H:%M:%S.%fZ").strftime("%Y-%m-%d %H:%M:%S")
        check_out = datetime.strptime(entry["checkOut"], "%Y-%m-%dT%H:%M:%S.%fZ").strftime("%Y-%m-%d %H:%M:%S")

        cursor.execute(
            """
            INSERT INTO bookings_info 
            (_id, platform, platform_id, listing_id, confirmation_code, check_in, check_out, 
            listing_title,account_id, guest_id,guest_name, commission)
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
            """,
            (
                entry["_id"],
                entry["integration"]["platform"],
                entry["integration"]["_id"],
                entry["listingId"],
                entry["confirmationCode"],
                check_in,
                check_out,
                entry["listing"]["title"],
                entry["accountId"],
                entry["guest"]["_id"],
                entry["guest"]["fullName"],
                entry["accounting"]["analytics"]["commission"],
            ),
        )

    conn.commit()
    cursor.close()
    conn.close()


# Load Dataset
def load_dataset(file_path):
    with open(file_path, "r") as f:
        data = json.load(f)  

    print(type(data))
    print(data["results"][0])
    return data["results"]


# Main Execution
data = load_dataset(r"C:\Users\ARYAN\OneDrive\Desktop\Neovis_Chatbot_Int\LLM-to-SQL\bookings.json") 
create_database_and_table(data)
