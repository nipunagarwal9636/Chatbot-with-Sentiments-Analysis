from src.chatbot.chatbot import Chatbot

def main():
    bot = Chatbot()
    print("Chatbot with Sentiments analysis(type 'exit' or 'quit' to finish)")
    while True:
        try:
            user_input = input("You: ").strip()
        except (KeyboardInterrupt, EOFError):
            print("\nExiting...")
            break
        if not user_input:
            continue
        if user_input.lower() in ("exit", "quit"):
            break
        bot.add_user_message(user_input)
        response = bot.get_response(user_input)
        print("Bot:", response)
    print("\nConversation ended. Computing sentiment analysis...\n")
    report = bot.conversation_sentiment_report()
    print(report)

if __name__ == '__main__':
    main()