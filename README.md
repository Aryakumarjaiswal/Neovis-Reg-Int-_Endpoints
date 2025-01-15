# Registered-User-endpoint-Neovis-
# FastAPI Chatbot Backend

This project is a FastAPI-based chatbot backend designed to provide information about Neovis Consulting. It utilizes AI to enhance user interactions and offers features such as user registration, chat history management, and customer service transfer.

## Folder Structure


## Installation

1. Clone the repository:
   ```bash
   git clone <repository-url>
   cd FastAPI-Chatbot-Backend
   ```

2. Set up a virtual environment:
   ```bash
   python -m venv custo_venv
   source custo_venv/bin/activate  # On Windows use `custo_venv\Scripts\activate`
   ```

3. Install the required packages:
   ```bash
   pip install -r requirements.txt
   ```

4. Create a `.env` file in the root directory and add your environment variables, such as:
   ```
   LOG_FILE_PATH=customer_main.log
   GEMINI_API_KEY=your_api_key_here
   ```

## Usage

To run the FastAPI application, execute the following command:



You can access the API documentation at `http://127.0.0.1:8000/docs`.

## Features

- **User Registration**: Users can sign up and log in to access personalized chat features.
- **Chat Functionality**: The chatbot can answer queries about Neovis Consulting and transfer chats to customer service if needed.
- **Context Retrieval**: Utilizes ChromaDB to retrieve relevant context for user queries, enhancing response accuracy.
- **Logging**: All interactions and errors are logged for monitoring and debugging purposes.

## Future Improvements

- Enhance AI response quality and user experience.
- Implement a feedback loop for continuous AI training.
- Optimize database queries for better performance.

## Contributing

Contributions are welcome! Please open an issue or submit a pull request for any enhancements or bug fixes.

## License

This project is licensed under the MIT License. See the LICENSE file for more details.
