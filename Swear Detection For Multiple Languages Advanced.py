import random
import discord
from discord.ext import commands, tasks
import asyncio
from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline
import torch
from torch.utils.data import DataLoader, Dataset
import numpy as np
from sklearn.model_selection import train_test_split
import pandas as pd
import sqlite3
import warnings
import emoji

# Initialize sentiment analysis and swear word detection models
def initialize_models():
    sentiment_tokenizer = AutoTokenizer.from_pretrained("nlptown/bert-base-multilingual-uncased-sentiment")
    sentiment_model = AutoModelForSequenceClassification.from_pretrained("nlptown/bert-base-multilingual-uncased-sentiment")
    swear_word_detector = pipeline('text-classification', model='finiteautomata/bertweet-base-sentiment-analysis')
    xlnet_tokenizer = AutoTokenizer.from_pretrained("minh21/XLNet-Reddit-Sentiment-Analysis")
    xlnet_model = AutoModelForSequenceClassification.from_pretrained("minh21/XLNet-Reddit-Sentiment-Analysis")
    return sentiment_tokenizer, sentiment_model, swear_word_detector, xlnet_tokenizer, xlnet_model

# Custom dataset class for sentiment analysis
class SentimentDataset(Dataset):
    def __init__(self, texts, labels):
        self.texts = texts
        self.labels = labels

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = self.texts[idx]
        label = self.labels[idx]
        return text, label

# Perform sentiment analysis on text using BERT-based model
def perform_bert_sentiment_analysis(sentiment_tokenizer, sentiment_model, text: str, device) -> int:
    inputs = sentiment_tokenizer.encode_plus(
        text,
        add_special_tokens=True,
        return_tensors="pt"
    ).to(device)
    outputs = sentiment_model(**inputs)
    predictions = torch.argmax(outputs.logits, dim=1)
    return predictions.item()

# Perform sentiment analysis on text using XLNet-based model
def perform_xlnet_sentiment_analysis(xlnet_tokenizer, xlnet_model, text: str, device) -> int:
    inputs = xlnet_tokenizer.encode_plus(
        text,
        add_special_tokens=True,
        return_tensors="pt"
    ).to(device)
    outputs = xlnet_model(**inputs)
    predictions = torch.argmax(outputs.logits, dim=1)
    return predictions.item()

# Detect swear words in text
def detect_swear_words(swear_word_detector, text: str) -> bool:
    result = swear_word_detector(text)
    return result[0]['label'] == 'SWEAR'

# Train a deep learning model for sentiment analysis
def train_sentiment_model(sentiment_tokenizer, sentiment_model, xlnet_tokenizer, xlnet_model, train_dataset, val_dataset, epochs=10, batch_size=32):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    sentiment_model.to(device)
    xlnet_model.to(device)

    # Align the output sizes of the BERT and XLNet models
    sentiment_model.classifier = torch.nn.Linear(sentiment_model.config.hidden_size, xlnet_model.config.num_labels).to(device)
    xlnet_model.classifier = torch.nn.Linear(xlnet_model.config.hidden_size, xlnet_model.config.num_labels).to(device)

    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    optimizer = torch.optim.Adam(list(sentiment_model.parameters()) + list(xlnet_model.parameters()), lr=1e-5)
    criterion = torch.nn.CrossEntropyLoss()

    best_val_loss = float('inf')

    for epoch in range(epochs):
        sentiment_model.train()
        xlnet_model.train()
        train_loss = 0.0
        for texts, labels in train_dataloader:
            optimizer.zero_grad()
            bert_inputs = sentiment_tokenizer.batch_encode_plus(texts, return_tensors="pt", padding=True).to(device)
            bert_outputs = sentiment_model(**bert_inputs)
            xlnet_inputs = xlnet_tokenizer.batch_encode_plus(texts, return_tensors="pt", padding=True).to(device)
            xlnet_outputs = xlnet_model(**xlnet_inputs)
            combined_outputs = (bert_outputs.logits + xlnet_outputs.logits) / 2
            labels = torch.tensor(labels).to(device)
            loss = criterion(combined_outputs, labels)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

        sentiment_model.eval()
        xlnet_model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for texts, labels in val_dataloader:
                bert_inputs = sentiment_tokenizer.batch_encode_plus(texts, return_tensors="pt", padding=True).to(device)
                bert_outputs = sentiment_model(**bert_inputs)
                xlnet_inputs = xlnet_tokenizer.batch_encode_plus(texts, return_tensors="pt", padding=True).to(device)
                xlnet_outputs = xlnet_model(**xlnet_inputs)
                combined_outputs = (bert_outputs.logits + xlnet_outputs.logits) / 2
                labels = torch.tensor(labels).to(device)
                loss = criterion(combined_outputs, labels)
                val_loss += loss.item()

        print(f"Epoch {epoch+1}/{epochs}, Train Loss: {train_loss/len(train_dataloader)}, Val Loss: {val_loss/len(val_dataloader)}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(sentiment_model.state_dict(), 'best_bert_sentiment_model.pth')
            torch.save(xlnet_model.state_dict(), 'best_xlnet_sentiment_model.pth')

# Load and preprocess the dataset
def load_and_preprocess_dataset():
    # Load data from the database
    conn = sqlite3.connect('sentiment_analysis.db')
    c = conn.cursor()
    c.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='sentiment_data'")
    table_exists = c.fetchone()
    if table_exists:
        c.execute("SELECT message, sentiment FROM sentiment_data")
        data = c.fetchall()
        texts = [row[0] for row in data]
        labels = [row[1] for row in data]
    else:
        texts = ["This is a great movie!", "I hate this product.", "The food was okay."]
        labels = [4, 1, 3]
    conn.close()
    return texts, labels

# Split the dataset into training and validation sets
def split_dataset(dataset):
    texts, labels = dataset
    train_texts, val_texts, train_labels, val_labels = train_test_split(texts, labels, test_size=0.2, random_state=42)
    return train_texts, train_labels, val_texts, val_labels

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

        sentiment_tokenizer, sentiment_model, swear_word_detector, xlnet_tokenizer, xlnet_model = initialize_models()
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        sentiment_model.to(device)
        xlnet_model.to(device)

        bert_sentiment = perform_bert_sentiment_analysis(sentiment_tokenizer, sentiment_model, message.content, device)
        xlnet_sentiment = perform_xlnet_sentiment_analysis(xlnet_tokenizer, xlnet_model, message.content, device)
        combined_sentiment = (bert_sentiment + xlnet_sentiment) // 2
        is_swear = detect_swear_words(swear_word_detector, message.content)
        save_message_to_database(message.author.id, message.content, combined_sentiment, is_swear)

        if combined_sentiment <= 2 or is_swear:
            await message.delete()
            warning_message = await message.channel.send(f'{message.author.mention}, please refrain from using inappropriate language.')
            await asyncio.sleep(10)
            await warning_message.delete()

        await bot.process_commands(message)

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

    # Check if the 'is_swear' column exists
    c.execute("PRAGMA table_info(sentiment_data)")
    columns = [column[1] for column in c.fetchall()]
    if 'is_swear' not in columns:
        # Add the 'is_swear' column if it doesn't exist
        c.execute('''ALTER TABLE sentiment_data ADD COLUMN is_swear BOOLEAN''')

    conn.commit()
    conn.close()

def save_message_to_database(user_id, message, sentiment, is_swear):
    conn = sqlite3.connect('sentiment_analysis.db')
    c = conn.cursor()
    c.execute("INSERT INTO sentiment_data (user_id, message, sentiment, is_swear) VALUES (?, ?, ?, ?)", (user_id, message, sentiment, int(is_swear)))
    conn.commit()
    conn.close()

# Replace 'YOUR_BOT_TOKEN' with your actual bot token
if __name__ == "__main__":
    # Load and preprocess the dataset
    dataset = load_and_preprocess_dataset()
    train_texts, train_labels, val_texts, val_labels = split_dataset(dataset)

    # Train a deep learning model for sentiment analysis
    sentiment_tokenizer, sentiment_model, swear_word_detector, xlnet_tokenizer, xlnet_model = initialize_models()
    train_dataset = SentimentDataset(train_texts, train_labels)
    val_dataset = SentimentDataset(val_texts, val_labels)
    train_sentiment_model(sentiment_tokenizer, sentiment_model, xlnet_tokenizer, xlnet_model, train_dataset, val_dataset)

    # Set up and run the Discord bot
    bot = setup_discord_bot()
    bot.run('MTEzMDg5NjA5NTczNjc3MDcwMg.Gl08cQ.XV4JplDM2vUqSf9lj1TCI1CBXB_8mhUdmTKIV8')