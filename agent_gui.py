import tkinter as tk
from tkinter import messagebox, ttk, filedialog
import babyagi_v3 as main
import speech_recognition as sr
import pyttsx3
import threading
import upload_documents
import os
from dotenv import load_dotenv
load_dotenv()


def on_submit():
    task_name = task_input.get()
    main.execute_main_loop(task_name)
    task_input.delete(0, tk.END)

def on_query():
    query = entry_query.get()
    if not query:
        messagebox.showerror("Error", "Please enter a query")
        return
    results = main.query_faiss_index(query, 5)  # Adjust the number of results (5) as needed
    result_text = "\n".join([f"{res[0]}: {res[1]:.2f}" for res in results])
    text_query_results.delete(1.0, tk.END)
    text_query_results.insert(tk.END, result_text)

def on_ask_question():
    system_message = entry_system_message.get()
    user_question = entry_user_question.get()
    faiss_results = int(entry_faiss_results.get())
    
    if not user_question:
        messagebox.showerror("Error", "Please enter a question")
        return
    
    answer = main.ask_direct_question(system_message, user_question)
    text_answer.delete(1.0, tk.END)
    text_answer.insert(tk.END, answer)

def on_upload():
    file_path = filedialog.askopenfilename()
    document_name = os.path.splitext(os.path.basename(file_path))[0]
    upload_documents.process_file(file_path, document_name)
    messagebox.showinfo("Information", "File uploaded and embeddings added to the FAISS index")

def threaded_listen_for_wake_word(engine):
    thread = threading.Thread(target=listen_for_wake_word, args=(engine,))
    thread.daemon = True
    thread.start()

def init_engine():
    engine = pyttsx3.init()
    voices = engine.getProperty('voices')
    engine.setProperty('voice', voices[1].id)  # Select a voice (0 = male, 1 = female)
    engine.setProperty('rate', 150)  # Set speech rate
    return engine

def listen_for_wake_word(engine):
    recognizer = sr.Recognizer()
    microphone = sr.Microphone()

    while True:
        with microphone as source:
            print("Listening for wake word...")
            recognizer.adjust_for_ambient_noise(source)
            audio = recognizer.listen(source)

        try:
            transcription = recognizer.recognize_google(audio)
            print(f"Transcription: {transcription}")

            if 'Athena' in transcription:
                engine.say("What can I do for you today?")
                engine.runAndWait()

                with microphone as source:
                    print("Listening for command...")
                    recognizer.adjust_for_ambient_noise(source)
                    audio = recognizer.listen(source)

                try:
                    command = recognizer.recognize_google(audio)
                    print(f"Command: {command}")

                    # Set default system message if not provided in the GUI
                    system_message = entry_system_message.get()
                    if not system_message:
                        system_message = "You are an AI who assists with tasks."

                    answer = main.ask_direct_question(system_message, command)
                    print(f"Answer: {answer}")

                    engine.say(answer)
                    engine.runAndWait()

                except sr.UnknownValueError:
                    print("Could not understand the command")
                except sr.RequestError as e:
                    print(f"Error with the speech recognition service: {e}")

        except sr.UnknownValueError:
            print("Could not understand the wake word")
        except sr.RequestError as e:
            print(f"Error with the speech recognition service: {e}")

def change_voice():
    global voice_id
    voice_id = (voice_id + 1) % len(voices)
    engine.setProperty('voice', voices[voice_id].id)

root = tk.Tk()
root.title("Agent GUI")
root.geometry("1400x800")

app_style = ttk.Style()

notebook = ttk.Notebook(root)
frame_main = ttk.Frame(notebook)
frame_questions = ttk.Frame(notebook)
frame_upload = ttk.Frame(notebook)

notebook.add(frame_main, text="Main")
notebook.add(frame_questions, text="Ask Questions")
notebook.add(frame_upload, text="Upload Documents")

notebook.pack(expand=1, fill="both")

label_upload = ttk.Label(frame_upload, text="Upload a document:")
label_upload.grid(column=0, row=0, padx=10, pady=10, sticky="w")

btn_upload = ttk.Button(frame_upload, text="Browse", command=on_upload)
btn_upload.grid(column=1, row=0, padx=10, pady=10)

# Initialize the text-to-speech engine
engine = init_engine()
voices = engine.getProperty('voices')
voice_id = 0

# Start listening for the wake word in a separate thread
root.after(100, lambda: threaded_listen_for_wake_word(engine))

# Main tab
label_objective = tk.Label(frame_main, text="Enter new objective (leave empty to use the current objective):")
label_objective.grid(row=0, column=0, padx=(10, 0), pady=10, sticky=tk.W)

entry_objective = tk.Entry(frame_main, width=100)
entry_objective.grid(row=0, column=1, padx=(10, 0), pady=10)

btn_update_objective = tk.Button(frame_main, text="Update Objective", command=on_submit)
btn_update_objective.grid(row=0, column=2, padx=(10, 0), pady=10)

label_query = tk.Label(frame_main, text="Enter query:")
label_query.grid(row=1, column=0, padx=(10, 0), pady=10, sticky=tk.W)

entry_query = tk.Entry(frame_main, width=100)
entry_query.grid(row=1, column=1, padx=(10, 0), pady=10)

btn_execute_query = tk.Button(frame_main, text="Execute Query", command=on_query)
btn_execute_query.grid(row=1, column=2, padx=(10, 0), pady=10)

label_query_results = tk.Label(frame_main, text="Query Results:")
label_query_results.grid(row=2, column=0, padx=(10, 0), pady=10, sticky=tk.W)

text_query_results = tk.Text(frame_main, width=100, height=5)
text_query_results.grid(row=2, column=1, padx=(10, 0), pady=10)

# Ask Questions tab
label_system_message = tk.Label(frame_questions, text="System message:")
label_system_message.grid(row=0, column=0, padx=(10, 0), pady=10, sticky=tk.W)

entry_system_message = tk.Entry(frame_questions, width=100)
entry_system_message.grid(row=0, column=1, padx=(10, 0), pady=10)

label_user_question = tk.Label(frame_questions, text="User question:")
label_user_question.grid(row=1, column=0, padx=(10, 0), pady=10, sticky=tk.W)

entry_user_question = tk.Entry(frame_questions, width=100)
entry_user_question.grid(row=1, column=1, padx=(10, 0), pady=10)

label_faiss_results = tk.Label(frame_questions, text="Number of FAISS results:")
label_faiss_results.grid(row=2, column=0, padx=(10, 0), pady=10, sticky=tk.W)

entry_faiss_results = tk.Entry(frame_questions, width=10)
entry_faiss_results.grid(row=2, column=1, padx=(10, 0), pady=10, sticky=tk.W)

btn_ask_question = tk.Button(frame_questions, text="Ask Question", command=on_ask_question)
btn_ask_question.grid(row=3, column=1, padx=(10, 0), pady=10, sticky=tk.W)

label_answer = tk.Label(frame_questions, text="Answer:")
label_answer.grid(row=4, column=0, padx=(10, 0), pady=10, sticky=tk.W)

text_answer = tk.Text(frame_questions, width=100, height=5)
text_answer.grid(row=4, column=1, padx=(10, 0), pady=10)

# Add voice selection button to the GUI
btn_change_voice = tk.Button(root, text="Change Voice", command=change_voice)
btn_change_voice.pack(side=tk.BOTTOM, pady=10)

root.mainloop()





