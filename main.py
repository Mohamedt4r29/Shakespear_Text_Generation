'''
Discuss each import and its work in the program:

- asyncio: Provides infrastructure for writing single-threaded concurrent code using coroutines.
- tkinter: GUI library for creating graphical user interfaces.
- Entry, Frame, Text, Scrollbar, Label, Button, Scale, Canvas, CENTER, messagebox: Various tkinter components for building the GUI.
- threading: Provides threading support for executing multiple threads concurrently.
- random: Generates pseudo-random numbers.
- numpy: Library for numerical operations.
- pyttsx3.engine.Engine: Text-to-speech library for Python.
- tensorflow, Sequential, LSTM, Dense, Activation, RMSprop: Machine learning library for building and training neural networks.
- pygame: Cross-platform set of Python modules designed for writing video games.
- pyttsx3: Text-to-speech library for Python.
- os: Provides a way to interact with the operating system.
- Image, ImageTk: Classes from the Python Imaging Library (PIL) for image processing.
- Canvas, NW: tkinter components for canvas and positioning.
- threading, queue: Support for multithreaded programming.
- spacy: Natural language processing library.
- pyperclip: Module for accessing the clipboard.
- filedialog: tkinter component for opening and saving files.
- pytube: Library for downloading YouTube videos.
- moviepy.editor.VideoFileClip: Library for video editing.
- openai: OpenAI GPT model for natural language processing.
'''
import asyncio
import tkinter as tk
from tkinter import Entry, Frame, Text, Scrollbar, Label, Button, Scale, Canvas, CENTER, messagebox
import threading
import random
from tkinter import messagebox
import numpy as np
from pyttsx3.engine import Engine
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Activation
from tensorflow.keras.optimizers import RMSprop
import pygame
import pyttsx3
import os
from PIL import Image, ImageTk
from tkinter import Canvas
from tkinter import NW
from PIL import Image, Image
import threading
import queue
import spacy
from PIL import Image, ImageTk
import pyperclip
from tkinter import filedialog
from pytube import YouTube
from moviepy.editor import VideoFileClip
import openai

'''
Class TextGenerator:

This class is responsible for generating text using a character-level LSTM model.

- __init__(self, file_url, seq_length=40, step_size=3): Initializes the TextGenerator object with parameters.
- load_text(self): Loads the text from the specified file URL and performs necessary preprocessing.
- process_text(self): Processes the loaded text to create character-to-index and index-to-character mappings.
- build_model(self): Builds an LSTM model for text generation.
- train_model(self, x, y, batch_size=256, epochs=4): Trains the LSTM model with the provided training data.
- save_model(self, model_name='textgenerator.model'): Saves the trained model to a file.
- load_model(self, model_name='textgenerator.model'): Loads a pre-trained model from a file.
- sample(self, preds, temperature=1.0): Samples the next character based on the model predictions.
- generate_text(self, length, temperature): Generates text of the specified length with the given temperature.
- generate_training_data(self): Placeholder method for generating training data (to be implemented).

Note: The class automatically checks for the existence of a pre-trained model file and either loads it or trains a new model if not found.
'''
class TextGenerator:
    def __init__(self, file_url, seq_length=40, step_size=3):                     
        self.file_url = file_url
        self.seq_length = seq_length
        self.step_size = step_size
        self.text = None
        self.characters = None
        self.char_to_index = None
        self.index_to_char = None
        self.model = None

        # Check if the model file exists
        import os
        model_file = 'textgenerator.model'
        if os.path.exists(model_file):
            self.load_model()
        else:
            self.load_text()
            self.process_text()
            if self.model is None:
                self.build_model()
                x, y = self.generate_training_data()
                self.train_model(x, y)
                self.save_model()

    def load_text(self):
        filepath = tf.keras.utils.get_file('shakespeare.txt', self.file_url)
        self.text = open(filepath, 'rb').read().decode(encoding='utf-8').lower()
        self.text = self.text[300000:800000]

    def process_text(self):
        self.characters = sorted(set(self.text))
        self.char_to_index = dict((c, i) for i, c in enumerate(self.characters))
        self.index_to_char = dict((i, c) for i, c in enumerate(self.characters))

    def build_model(self):
        model = Sequential()
        model.add(LSTM(128, input_shape=(self.seq_length, len(self.characters))))
        model.add(Dense(len(self.characters)))
        model.add(Activation('softmax'))

        optimizer = RMSprop(learning_rate=0.01)
        model.compile(loss='categorical_crossentropy', optimizer=optimizer)

        self.model = model

        return model

    def train_model(self, x, y, batch_size=256, epochs=4):
        self.model.fit(x, y, batch_size=batch_size, epochs=epochs)

    def save_model(self, model_name='textgenerator.model'):
        self.model.save(model_name)

    def load_model(self, model_name='textgenerator.model'):
        self.model = tf.keras.models.load_model(model_name)

    def sample(self, preds, temperature=1.0):
        preds = np.asarray(preds).astype('float64')
        preds = np.log(preds) / temperature
        exp_preds = np.exp(preds)
        preds = exp_preds / np.sum(exp_preds)
        probas = np.random.multinomial(1, preds, 1)
        return np.argmax(probas)
    

    def generate_text(self, length, temperature):
        start_index = random.randint(0, len(self.text) - self.seq_length - 1)
        generated = ''
        sentence = self.text[start_index: start_index + self.seq_length]
        generated += sentence
        for _ in range(length):
            x = np.zeros((1, self.seq_length, len(self.characters)))
            for t, character in enumerate(sentence):
                x[0, t, self.char_to_index[character]] = 1

            predictions = self.model.predict(x, verbose=0)[0]
            next_index = self.sample(predictions, temperature)
            next_character = self.index_to_char[next_index]

            generated += next_character
            sentence = sentence[1:] + next_character
        return generated

    def generate_training_data(self):
        x, y = None, None
        return x, y
         

'''
Class TextGeneratorGUI:

This class represents the graphical user interface (GUI) for the Text Generator application.

- __init__(self, master): Initializes the TextGeneratorGUI object with the specified master window.
- generate_text(self): Invoked when the "Generate Text" button is clicked, triggers the text generation process.
- copy_text(self): Invoked when the "Copy Text" button is clicked, copies the generated text to the clipboard.
- clear_text(self): Invoked when the "Clear Text" button is clicked, clears the generated text from the display.
- toggle_music(self): Invoked when the "Play Music" button is clicked, toggles between playing and pausing music.
- toggle_poem(self): Invoked when the "Read Poem" button is clicked, toggles between reading and stopping poem reading.
- set_volume(self, value): Invoked when the volume scale is adjusted, sets the volume of the music.
- change_music(self): Invoked when the "Change Music" button is clicked, changes the currently playing music.
- download_music(self): Invoked when the "Download" button is clicked, downloads music from the specified YouTube link.
- complete_user_input(self): Invoked when the "Complete Input" button is clicked, completes user input using the Text Generator model.

Note: The class initializes the GUI components, handles button clicks, and interacts with the TextGenerator model.
'''
class TextGeneratorGUI:
    def __init__(self, master):
        self.master = master
        master.title("Text Generator GUI")
        self.tts_thread = None
        self.tts_queue = queue.Queue()  
        self.nlp = spacy.load('en_core_web_sm')
        self.text_generator = TextGenerator('https://storage.googleapis.com/download.tensorflow.org/data/shakespeare.txt')
        self.music_playing = False
        self.poem_reading = False
        self.engine = pyttsx3.init()
        self.engine.setProperty('rate', 150)
        self.engine.setProperty('volume', 1.0)
        # Initialize pygame.mixer and load the sound file
        pygame.mixer.init()
        self.hello_sound = pygame.mixer.Sound("hello_sound.wav")  # Replace with the actual path to your sound file
        # Play the hello sound automatically when the app is opened
        self.play_hello_sound()


        self.bg_image = Image.open("image3.jpg")

        if hasattr(Image, 'ANTIALIAS'):
            self.bg_image = self.bg_image.resize((master.winfo_screenwidth(), master.winfo_screenheight()), Image.ANTIALIAS)
        else:
            self.bg_image = self.bg_image.resize((master.winfo_screenwidth(), master.winfo_screenheight()))

        self.bg_photo = ImageTk.PhotoImage(self.bg_image)

        self.canvas = tk.Canvas(master, width=master.winfo_screenwidth(), height=master.winfo_screenheight())
        self.canvas.create_image(0, 0, anchor=NW, image=self.bg_photo)
        self.canvas.pack()

        self.inner_frame = Frame(self.canvas, bg=None)
        self.inner_frame.pack()

        self.output_text = Text(self.inner_frame, wrap='word', height=10, width=60, font=('Helvetica', 12), bg=None, fg='black')
        self.output_text.pack(pady=10, expand=True, fill='both')
        self.output_text.tag_configure("center", justify="center")
        scroll = Scrollbar(self.inner_frame, command=self.output_text.yview)
        self.output_text.config(yscrollcommand=scroll.set)
        scroll.pack(side='right', fill='y')

        self.processing_label = Label(self.inner_frame, text="Processing...", font=("Helvetica", 12), fg="blue", bg='red')
        self.processing_label.pack_forget()

        self.bottom_margin_frame = Frame(self.inner_frame, bg='white')
        self.bottom_margin_frame.pack(side='bottom', fill='both', pady=10)

        copyright_label = Label(self.bottom_margin_frame, text="Copyright © 2024 Mohamed_GPT contact the developer 246661@vutbr.cz", font=("Helvetica", 10), fg="black", bg='white')
        copyright_label.pack(side='bottom')

        text_frame = Frame(self.inner_frame, bg='black')
        text_frame.pack(side='top', pady=10)


        generate_button = Button(text_frame, text="Generate Text", command=self.generate_text, bg='white', bd=4, relief=tk.RAISED, highlightcolor='yellow')
        generate_button.pack(side=tk.LEFT, padx=5)

        # Draw a red line between buttons
        self.draw_line(text_frame)

        copy_button = Button(text_frame, text="Copy Text", command=self.copy_text, bg='white', bd=4, relief=tk.RAISED, highlightcolor='yellow')
        copy_button.pack(side=tk.LEFT, padx=5)

        # Draw a red line between buttons
        self.draw_line(text_frame)

        # Add a "Clear Text" button
        clear_text_button = Button(text_frame, text="Clear Text", command=self.clear_text, bg='white', bd=4, relief=tk.RAISED, highlightcolor='yellow')
        clear_text_button.pack(side=tk.LEFT, padx=5)

        

        music_frame = Frame(self.inner_frame, bg='black')
        music_frame.pack(side='top', pady=10)

        music_button = Button(music_frame, text="Play Music", command=self.toggle_music, bg='blue', bd=4, relief=tk.RAISED, highlightcolor='yellow')
        music_button.pack(side=tk.LEFT, padx=5)
        self.music_button = music_button

        # Add a vertical line between "Play Music" and "Read Poem"
        self.draw_line(music_frame, orient=tk.VERTICAL)



        read_poem_button = Button(music_frame, text="Read Poem", command=self.toggle_poem, bg='blue', bd=4, relief=tk.RAISED, highlightcolor='yellow')
        read_poem_button.pack(side=tk.LEFT, padx=5)
        self.read_poem_button = read_poem_button

        volume_scale = Scale(music_frame, label="Volume", from_=0, to=100, orient="horizontal", command=self.set_volume)
        volume_scale.pack(side=tk.LEFT, padx=5)
        volume_scale.set(50)


        change_music_button = Button(music_frame, text="Change Music", command=self.change_music, bg='white', bd=4, relief=tk.RAISED, highlightcolor='yellow')
        change_music_button.pack(side=tk.LEFT, padx=5)
        self.change_music_button = change_music_button


        youtube_frame = Frame(self.inner_frame, bg='black')
        youtube_frame.pack(side='top', pady=10)

        youtube_label = Label(youtube_frame, text="Enter YouTube Link:", font=("Helvetica", 12), fg="black", bg='white')
        youtube_label.pack(side=tk.LEFT, padx=5)

        self.youtube_entry = Entry(youtube_frame, width=30, font=("Helvetica", 12))
        self.youtube_entry.pack(side=tk.LEFT, padx=5)

        download_button = Button(youtube_frame, text="Download", command=self.download_music, bg='white', bd=4, relief=tk.RAISED, highlightcolor='yellow')
        download_button.pack(side=tk.LEFT, padx=5)

        self.inner_frame.place(relx=0.5, rely=0.5, anchor=CENTER)

        # Add entry widget for user input
        user_input_frame = Frame(self.inner_frame, bg='black')
        user_input_frame.pack(side='top', pady=10)
        user_input_label = Label(user_input_frame, text="Enter your text:", font=("Helvetica", 12), fg="black", bg='white')
        user_input_label.pack(side=tk.LEFT, padx=5)
        self.user_input_entry = Entry(user_input_frame, width=40, font=("Helvetica", 12))
        self.user_input_entry.pack(side=tk.LEFT, padx=5)

        complete_button = Button(user_input_frame, text="Complete Input", command=self.complete_user_input, bg='white', bd=4, relief=tk.RAISED, highlightcolor='yellow')
        complete_button.pack(side=tk.LEFT, padx=5)



  
    '''
      Methods related to sound and visuals:

      - play_hello_sound(self): Plays the hello sound.
      - draw_line(self, parent_frame, orient=tk.VERTICAL): Draws a line on the specified frame, either horizontally or vertically.
      - clear_text(self): Clears the generated text in the output area.
      - complete_user_input(self): Completes user input using OpenAI GPT-3 asynchronously.
      - complete_with_gpt3_async(self, input_text): Asynchronously completes user input using OpenAI GPT-3.
      - download_youtube_video(self, youtube_link): Downloads a YouTube video as an MP3 file.
      - copy_text(self): Copies the generated text to the clipboard.
      - toggle_music(self): Toggles between playing and pausing background music.
      - show_processing_message(self): Displays a processing message.
      - hide_processing_message(self): Hides the processing message.
      - generate_text(self): Initiates the text generation process.
      - generate_text_thread(self): Runs the text generation process in a separate thread.
      - filter_generated_text(self, generated_text): Filters out incomplete or nonsensical lines from generated text.
      - is_semantically_valid(self, sentence): Checks if a sentence is semantically valid based on spaCy similarity.
      - set_volume(self, value): Sets the volume of the background music.
      - volume_up(self): Increases the volume of the background music.
      - volume_down(self): Decreases the volume of the background music.
      - change_music(self): Prompts the user to select a new music file and plays it.
      - update_music_button_state(self): Updates the state and text of the music button.
      - stop_music(self): Stops the background music.
      - update_poem_button_state(self): Updates the state and text of the poem button.
      - download_music(self): Downloads music from the specified YouTube link.
      - toggle_poem(self): Toggles between reading and stopping poem reading.
      - read_poem(self): Reads the generated poem using text-to-speech.
      - stop_poem(self): Stops the poem reading.
      - update_poem_button_state(self): Updates the state and text of the poem button.
      - text_to_speech_thread(self, engine, poem): Runs text-to-speech for the generated poem in a separate thread.
      '''




    def play_hello_sound(self):
        # Method to play the hello sound
        self.hello_sound.play()

    
    def draw_line(self, parent_frame, orient=tk.VERTICAL):
        # Create a canvas for drawing
        line_canvas = Canvas(parent_frame, width=2, height=40, bg='white', highlightthickness=0)
        line_canvas.pack(side=tk.LEFT, padx=5)

        # Draw a red line horizontally or vertically
        if orient == tk.HORIZONTAL:
            line_canvas.create_line(0, 1, 2, 1, fill='red', width=2)
        else:
            line_canvas.create_line(1, 0, 1, 40, fill='red', width=2)

        # Draw a red line
        line_canvas.create_line(1, 0, 1, 40, fill='red', width=2)

    def clear_text(self):
        # Clear the generated text in the output area
        self.output_text.delete(1.0, tk.END)
    
    
    def complete_user_input(self):
        user_input = self.user_input_entry.get()

        # Check if the user input is long enough
        if len(user_input) < 10:
            messagebox.showwarning("Input Warning", "Please provide a longer input for better completion.")
            return

        # Perform text completion using OpenAI GPT-3 asynchronously
        asyncio.run(self.complete_with_gpt3_async(user_input))

    async def complete_with_gpt3_async(self, input_text):
        try:
            # Set up OpenAI API key
            openai.api_key = 'your-api-key'

            # Make an asynchronous request to GPT-3 for text completion
            response = await asyncio.to_thread(
                openai.Completion.create,
                engine="text-davinci-002",
                prompt=input_text,
                max_tokens=150,
                temperature=0.7,
            )

            # Extract the generated text from GPT-3 response
            completion_text = response.choices[0].text.strip()

            # Display the completion in the text area
            self.output_text.insert(tk.END, "\n\nUser Input Completion:\n" + completion_text, "center")

            # Clear the user input entry
            self.user_input_entry.delete(0, tk.END)

        except Exception as e:
            print(f"An error occurred during GPT-3 completion: {e}")
    
    def download_youtube_video(self, youtube_link):
        # Get the absolute path of the script's directory
        script_directory = os.path.dirname(os.path.abspath(__file__))

        # Open a file dialog to let the user choose the save location (as a directory)
        save_path = filedialog.askdirectory(initialdir=script_directory)

        # If the user cancels the dialog, do nothing
        if not save_path:
            return

        # Download the YouTube video
        try:
            yt = YouTube(youtube_link)
            yt_stream = yt.streams.filter(only_audio=True).first()

            # Specify the output file path (MP4 format)
            mp4_output_file_path = os.path.join(script_directory, 'downloaded_music.mp4')

            # Download as MP4
            yt_stream.download(mp4_output_file_path)

            # Specify the output file path (MP3 format)
            mp3_output_file_path = os.path.join(save_path, 'downloaded_music.mp3')

            # Convert MP4 to MP3
            clip = VideoFileClip(mp4_output_file_path)
            clip.audio.write_audiofile(mp3_output_file_path, codec='mp3')

            # Delete the temporary MP4 file
            os.remove(mp4_output_file_path)

            messagebox.showinfo("Download Successful", "Music downloaded successfully!")
        except Exception as e:
            messagebox.showerror("Error", f"An error occurred: {e}")

    def copy_text(self):
        generated_text = self.output_text.get("1.0", tk.END)
        pyperclip.copy(generated_text)
        messagebox.showinfo("Copy Successful", "Text copied to clipboard!")

    def toggle_music(self):
        if not self.music_playing:
            self.play_background_music()
        else:
            self.stop_music()

    def show_processing_message(self):
        self.processing_label.pack()

    def hide_processing_message(self):
        self.processing_label.pack_forget()

    def generate_text(self):
        self.show_processing_message()
        thread = threading.Thread(target=self.generate_text_thread)
        thread.start()

    def generate_text_thread(self):
        try:
            self.text_generator.load_text()
            self.text_generator.process_text()
            generated_text = ''
            generated_text += '----------0.2--------\n'
            generated_text += self.text_generator.generate_text(300, 0.2) + '\n'
            generated_text += '----------0.4--------\n'
            generated_text += self.text_generator.generate_text(300, 0.4) + '\n'
            generated_text += '----------0.6--------\n'
            generated_text += self.text_generator.generate_text(300, 0.6) + '\n'
            generated_text += '----------0.8--------\n'
            generated_text += self.text_generator.generate_text(300, 0.8) + '\n'
            generated_text += '----------1.0--------\n'
            generated_text += self.text_generator.generate_text(300, 1.0) + '\n'

            # Apply post-processing to filter out incomplete or nonsensical lines
            filtered_text = self.filter_generated_text(generated_text)
    
            self.master.after(0, self.hide_processing_message)
            self.master.after(0, lambda: self.output_text.insert(tk.END, filtered_text, "center"))

        except Exception as e:
            print(f"An error occurred: {e}")    # Add additional error handling or logging as needed

        # Apply post-processing to filter out incomplete or nonsensical lines. Additionally, use NLP to improve grammar and punctuation.

        filtered_text = self.filter_generated_text(generated_text)

        self.master.after(0, self.hide_processing_message)
        self.master.after(0, lambda: self.output_text.insert(tk.END, filtered_text, "center"))


    def filter_generated_text(self, generated_text):
        lines = generated_text.split('\n')
        # Adjust filtering criteria based on your preference
        filtered_lines = [line for line in lines if len(line) > 20 and line.endswith(('.', '!', '?'))]
        
        # Apply semantic analysis to filter out sentences with low semantic similarity
        filtered_lines = [line for line in filtered_lines if self.is_semantically_valid(line)]

        return '\n'.join(filtered_lines)

    def is_semantically_valid(self, sentence):
        # Calculate semantic similarity using spaCy
        training_data = ' '.join(self.text_generator.text.split('\n')[:100])  # Use a portion of the training data for comparison
        gen_doc = self.nlp(sentence)
        similarity_scores = [gen_doc.similarity(self.nlp(train_sentence)) for train_sentence in training_data.split('\n')]

        # Threshold for semantic similarity
        threshold = 0.5  # Adjust as needed
        return max(similarity_scores) > threshold

    def set_volume(self, value):
        volume = float(value) / 100.0
        pygame.mixer.music.set_volume(volume)

    def volume_up(self):
        current_volume = pygame.mixer.music.get_volume()
        new_volume = min(current_volume + 0.1, 1.0)
        pygame.mixer.music.set_volume(new_volume)

    def volume_down(self):
        current_volume = pygame.mixer.music.get_volume()
        new_volume = max(current_volume - 0.1, 0.0)
        pygame.mixer.music.set_volume(new_volume)

    def change_music(self):
        # Stop the music if it's playing
        if self.music_playing:
            self.stop_music()

        # Prompt the user to select a new music file
        file_path = filedialog.askopenfilename(title="Select Music File", filetypes=[("MP3 files", "*.mp3")])

        # Load and play the new music file
        if file_path:
            pygame.mixer.music.load(file_path)
            pygame.mixer.music.play()
            self.music_playing = True
            self.update_music_button_state()

    def copy_text(self):
        generated_text = self.output_text.get("1.0", tk.END)
        pyperclip.copy(generated_text)
        messagebox.showinfo("Copy Successful", "Text copied to clipboard!")

    def toggle_music(self):
        if not self.music_playing:
            self.play_background_music()
        else:
            self.stop_music()

    def play_background_music(self):
        if not pygame.mixer.get_init():
            pygame.mixer.init(frequency=22050, size=-16, channels=2, buffer=4096)

        # Get the absolute path of the script's directory
        script_directory = os.path.dirname(os.path.abspath(__file__))

        # Specify the relative path to the music file
        music_file_path = os.path.join(script_directory, 'Night Music Romanze.mp3')

        pygame.mixer.music.load(music_file_path)
        pygame.mixer.music.play()
        self.music_playing = True
        self.update_music_button_state()

    def stop_music(self):
        pygame.mixer.music.stop()
        self.music_playing = False
        self.update_music_button_state()

    def update_music_button_state(self):
        if self.music_playing:
            self.music_button.config(text="Stop Music", bg='red')
        else:
            self.music_button.config(text="Play Music", bg='blue')

    def download_music(self):
        # Stop the music if it's playing
        if self.music_playing:
            self.stop_music()

        # Get the YouTube link from the entry widget
        youtube_link = self.youtube_entry.get()

        # Download the YouTube video as an MP3 file
        if youtube_link:
            try:
                self.download_youtube_video(youtube_link)
                messagebox.showinfo("Download Successful", "Music downloaded successfully!")
            except Exception as e:
                messagebox.showerror("Error", f"An error occurred: {e}")

    def toggle_poem(self):
        if not self.poem_reading:
            self.read_poem()
            self.read_poem_button.config(text="Stop Poem", bg='red')
        else:
            self.stop_poem()
            self.read_poem_button.config(text="Read Poem", bg='blue')

    def read_poem(self):
        # Initialize the text-to-speech engine
        engine = pyttsx3.init()
        engine.setProperty('rate', 150)  # Speed of speech
        engine.setProperty('volume', 1.0)  # Volume level (0.0 to 1.0)

        # Get the generated poem from the Text widget
        poem = self.output_text.get("1.0", tk.END)

        # Clear the existing queue
        self.tts_queue.queue.clear()

        # Start a new thread for text-to-speech
        self.tts_thread = threading.Thread(target=self.text_to_speech_thread, args=(engine, poem))
        self.tts_thread.start()

        self.poem_reading = True
        self.read_poem_button.config(text="Stop Poem", bg='red')

    def stop_poem(self):
        # Stop the text-to-speech thread if it's running
        if self.tts_thread and self.tts_thread.is_alive():
            # Add logic to stop the poem reading
            print("Stopping poem reading...")
            self.engine.stop()  # Stop the text-to-speech engine
            self.tts_thread.join()  # Wait for the thread to finish
            print("Poem reading stopped.")
            # Clear the queue to avoid processing 'done' signal
            with self.tts_queue.mutex:
                self.tts_queue.queue.clear()

        # Ensure the text-to-speech engine is stopped
        self.engine.stop()

        self.poem_reading = False
        self.read_poem_button.config(text="Read Poem", bg='blue')


    def update_poem_button_state(self):
        if self.poem_reading:
            self.read_poem_button.config(text="Stop Poem", bg='red')
        else:
            self.read_poem_button.config(text="Read Poem", bg='blue')


    def text_to_speech_thread(self, engine, poem):
        # Split the poem into lines
        poem_lines = poem.split('\n')

        # Read each line with pauses in between
        for line in poem_lines:
            # Skip empty lines
            if line.strip() == '':
                continue

            # Use the text-to-speech engine to read the line
            engine.say(line)

            # Wait for the speech to finish
            engine.runAndWait()

        # Signal the main thread that text-to-speech is complete
        self.tts_queue.put("done")


if __name__ == "__main__":
    root = tk.Tk()
    app = TextGeneratorGUI(root)
    root.mainloop()