import random
import discord
from discord.ext import commands, tasks
import asyncio
from transformers import pipeline, AutoTokenizer, AutoModelForSequenceClassification
import torch
from torch.utils.data import DataLoader, Dataset
import numpy as np
from sklearn.model_selection import train_test_split
import sqlite3

# Initialize sentiment analysis and swear word detection models
sentiment_tokenizer = AutoTokenizer.from_pretrained("nlptown/bert-base-multilingual-uncased-sentiment")
sentiment_model = AutoModelForSequenceClassification.from_pretrained("nlptown/bert-base-multilingual-uncased-sentiment")
swear_word_detector = pipeline('text-classification', model='finiteautomata/bertweet-base-sentiment-analysis')
roberta_sentiment_tokenizer = AutoTokenizer.from_pretrained("cardiffnlp/twitter-roberta-base-sentiment-latest")
roberta_sentiment_model = AutoModelForSequenceClassification.from_pretrained("cardiffnlp/twitter-roberta-base-sentiment-latest", ignore_mismatched_sizes=True)

# Custom dataset class for sentiment analysis
class SentimentDataset(Dataset):
    def __init__(self, texts, labels):
        self.texts, self.labels = texts, labels

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        return self.texts[idx], self.labels[idx]

# Perform sentiment analysis on text
def perform_sentiment_analysis(tokenizers, models, texts, device):
    bert_inputs = sentiment_tokenizer.batch_encode_plus(texts, return_tensors="pt", padding=True).to(device)
    bert_outputs = sentiment_model(**bert_inputs)
    roberta_inputs = roberta_sentiment_tokenizer.batch_encode_plus(texts, return_tensors="pt", padding=True).to(device)
    roberta_outputs = roberta_sentiment_model(**roberta_inputs)
    combined_outputs = torch.cat((bert_outputs.logits, roberta_outputs.logits), dim=1)
    print(f"Combined Outputs: {combined_outputs}")
    return torch.argmax(combined_outputs, dim=1).tolist()

# Detect swear words in text
def detect_swear_words(swear_word_detector, text):
    result = swear_word_detector(text)
    print(f"Swear Word Detection Result: {result}")
    return result[0]['label'] == 'SWEAR'

# Train a deep learning model for sentiment analysis
def train_sentiment_model(train_dataset, val_dataset, epochs=10, batch_size=32):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    sentiment_model.to(device)
    roberta_sentiment_model.to(device)

    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    optimizer = torch.optim.Adam(list(sentiment_model.parameters()) + list(roberta_sentiment_model.parameters()), lr=1e-5)
    criterion = torch.nn.CrossEntropyLoss()

    best_val_loss = float('inf')
    for epoch in range(epochs):
        train_loss, val_loss = 0.0, 0.0
        for texts, labels in train_dataloader:
            optimizer.zero_grad()
            combined_outputs = perform_sentiment_analysis([sentiment_tokenizer, roberta_sentiment_tokenizer], [sentiment_model, roberta_sentiment_model], texts, device)
            combined_outputs_tensor = torch.tensor(combined_outputs, dtype=torch.float32, requires_grad=True).to(device)
            labels_tensor = torch.tensor(labels, dtype=torch.float32).to(device)
            loss = criterion(combined_outputs_tensor, labels_tensor)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
        with torch.no_grad():
            for texts, labels in val_dataloader:
                combined_outputs = perform_sentiment_analysis([sentiment_tokenizer, roberta_sentiment_tokenizer], [sentiment_model, roberta_sentiment_model], texts, device)
                combined_outputs_tensor = torch.tensor(combined_outputs, dtype=torch.float32, requires_grad=True).to(device)
                labels_tensor = torch.tensor(labels, dtype=torch.float32).to(device)
                loss = criterion(combined_outputs_tensor, labels_tensor)
                val_loss += loss.item()
        print(f"Epoch {epoch+1}/{epochs}, Train Loss: {train_loss/len(train_dataloader)}, Val Loss: {val_loss/len(val_dataloader)}")
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(sentiment_model.state_dict(), 'best_bert_sentiment_model.pth')
            torch.save(roberta_sentiment_model.state_dict(), 'best_roberta_sentiment_model.pth')

# Load and preprocess the dataset
def load_and_preprocess_dataset():
    conn = sqlite3.connect('sentiment_analysis.db')
    c = conn.cursor()
    c.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='sentiment_data'")
    if c.fetchone():
        c.execute("SELECT message, sentiment, is_swear FROM sentiment_data")
        data = c.fetchall()
        texts, labels, is_swear = [row[0] for row in data], [row[1] for row in data], [row[2] for row in data]
    else:
        texts, labels, is_swear = [], [], []
    conn.close()
    return texts, labels, is_swear

# Split the dataset into training and validation sets
def split_dataset(dataset):
    texts, labels, is_swear = dataset
    train_texts, val_texts, train_labels, val_labels, train_is_swear, val_is_swear = train_test_split(texts, labels, is_swear, test_size=0.2, random_state=42)
    return train_texts, train_labels, val_texts, val_labels, train_is_swear, val_is_swear

# Discord bot setup
def setup_discord_bot():
    intents = discord.Intents.default()
    intents.message_content = True
    bot = commands.Bot(command_prefix='!', intents=intents)

    @bot.event
    async def on_ready():
        print(f'Bot is ready. Logged in as {bot.user}')
        update_status.start()
        create_database()

    @tasks.loop(minutes=1)
    async def update_status():
        await bot.change_presence(activity=discord.Activity(type=discord.ActivityType.watching, name=get_random_status()))

    @bot.event
    async def on_message(message):
        if message.author == bot.user:
            return

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        combined_sentiment = perform_sentiment_analysis([sentiment_tokenizer, roberta_sentiment_tokenizer], [sentiment_model, roberta_sentiment_model], [message.content], device)[0]
        print(f"Combined Sentiment: {combined_sentiment}")
        is_swear = detect_swear_words(swear_word_detector, message.content)
        print(f"Is Swear: {is_swear}")
        save_message_to_database(message.author.id, message.content, combined_sentiment, is_swear)

        if is_swear or combined_sentiment <= 2:
            try:
                await message.delete()
                warning_message = await message.channel.send(f'{message.author.mention}, please refrain from using inappropriate language.', delete_after=10)
            except discord.Forbidden:
                print(f"Failed to handle message from {message.author.name} due to insufficient permissions.")
        else:
            await bot.process_commands(message)

    @bot.command()
    async def ping(ctx):
        await ctx.send('Pong!')

    @bot.command()
    async def sentiment(ctx, *, text: str):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        combined_sentiment = perform_sentiment_analysis([sentiment_tokenizer, roberta_sentiment_tokenizer], [sentiment_model, roberta_sentiment_model], [text], device)[0]
        await ctx.send(f'The sentiment of the text is: {combined_sentiment}/5')

    return bot

def get_random_status():
    statuses = [
        "Analyzing messages...",
        "Keeping the chat clean!",
        "Sentiment detection in progress",
        "Swear word detection active",
    ]
    return random.choice(statuses)

def create_database():
    conn = sqlite3.connect('sentiment_analysis.db')
    c = conn.cursor()
    c.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='sentiment_data'")
    if not c.fetchone():
        c.execute('''CREATE TABLE sentiment_data
                     (user_id INTEGER, message TEXT, sentiment INTEGER, is_swear BOOLEAN)''')
    conn.commit()
    conn.close()

def save_message_to_database(user_id, message, sentiment, is_swear):
    conn = sqlite3.connect('sentiment_analysis.db')
    c = conn.cursor()
    c.execute("INSERT INTO sentiment_data (user_id, message, sentiment, is_swear) VALUES (?, ?, ?, ?)", (user_id, message, sentiment, is_swear))
    conn.commit()
    conn.close()

if __name__ == "__main__":
    dataset = load_and_preprocess_dataset()
    train_texts, train_labels, val_texts, val_labels, train_is_swear, val_is_swear = split_dataset(dataset)
    train_dataset = SentimentDataset(train_texts, train_labels)
    val_dataset = SentimentDataset(val_texts, val_labels)
    train_sentiment_model(train_dataset, val_dataset)
    bot = setup_discord_bot()
    bot.run('Your-Bot-Token-Here')
