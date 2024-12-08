# Data_Analytics_project
this is my data Analytics project using machine learning -my project is question answering system that main work is to provide the answers to the users
from flask import Flask, render_template, request, jsonify, send_file
import os
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from langchain.chains import RetrievalQA
from langchain.llms import HuggingFaceHub
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer
from scipy.stats import pearsonr, spearmanr
from sklearn.metrics import mean_squared_error
from fuzzywuzzy import fuzz
import numpy as np
import matplotlib.pyplot as plt
import re
from io import BytesIO
import torch
from diffusers import StableDiffusionPipeline
import cv2
import easyocr
from werkzeug.utils import secure_filename
from tensorflow.keras.applications.vgg16 import VGG16, preprocess_input
from tensorflow.keras.preprocessing import image
from tensorflow.keras.models import Model
import datetime



# Set your Hugging Face API token for access to the models
os.environ["HUGGINGFACEHUB_API_TOKEN"] = "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"

app = Flask(__name__)

# <------Define all routes here-------->

@app.route('/')
@app.route('/question-ans')
def question_ans():
    return render_template('question-ans.html')

#@app.route('/image-text')
#def imagetext():
#    return render_template('image-text.html')

@app.route('/text-image')
def textimage():
    return render_template('text-image.html')

#@app.route('/similar-image')
#def similarimage():
#    return render_template('similar-image.html')

@app.route('/about')
def about():
    return render_template('about.html')

@app.route('/contact_form')
def contact_form():
    return render_template('contact_form.html')

@app.route('/contacts', methods=['GET', 'POST'])
def contacts():
    if request.method == 'POST':
        name = request.form.get('name')
        email = request.form.get('email')
        message = request.form.get('message')
        print(f"Received message from {name} ({email}): {message}")

    return render_template('contacts.html')

# Predefined questions and answers
predefined_qa = {
    "hi": "Hello! How can I assist you today?",
    "hlo": "Hi there! What can I do for you?",
    "who are you": "I am a virtual assistant powered by AI. How can I help you?",
    "how are you": "I’m just a bunch of code, but thanks for asking! How about you?",
    "what is your name": "You can call me your assistant!",
    "bye": "Goodbye! Have a great day!"
}

# Function to extract only the helpful answer part from the response
def get_helpful_answer(response_text):
    match = re.search(r"Helpful Answer:\s*(.*)", response_text, re.DOTALL)
    if match:
        return match.group(1).strip()  # Return only the text after 'Helpful Answer:'
    return response_text.strip()  # Return the full response if no "Helpful Answer" section is found

# Define ground truth questions and answers (only for similarity calculation)
ground_truth = {
    "When and where was Virat Kohli born?": "Virat Kohli was born on November 5, 1988, in Delhi, India.",
    # Add more ground truth entries as needed...
}

# Initialize embeddings model globally
embeddings_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2")

# Extract text and chunks
def get_txt_text(txt_docs):
    text = ""
    for txt_file in txt_docs:
        with open(txt_file, 'r', encoding='utf-8') as file:
            text += file.read()
    return text

def get_text_chunks(text):
    text_splitter = CharacterTextSplitter(
        separator="\n",
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len
    )
    chunks = text_splitter.split_text(text)
    return chunks

def get_vectorstore(text_chunks):
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2")
    vectorstore = FAISS.from_texts(texts=text_chunks, embedding=embeddings)
    return vectorstore

def retrieval_qa_chain(db, return_source_documents=False):
    llm = HuggingFaceHub(repo_id="tiiuae/falcon-7b-instruct", model_kwargs={"temperature": 0.3, "max_new_tokens": 300})
    qa_chain = RetrievalQA.from_chain_type(llm=llm, chain_type='stuff', retriever=db,
                                           return_source_documents=return_source_documents)
    return qa_chain

# Load dataset and initialize model
path_to_txt = [r'C:\Users\user\Desktop\dataset_qa.txt']
raw_text = get_txt_text(path_to_txt)
text_chunks = get_text_chunks(raw_text)
vectorstore = get_vectorstore(text_chunks)
db = vectorstore.as_retriever(search_kwargs={'k': 5})
bot = retrieval_qa_chain(db, return_source_documents=False)

# Function to find the closest ground truth question
def find_closest_question(user_query, ground_truth):
    user_query_embedding = embeddings_model.embed_query(user_query)
    ground_truth_questions = list(ground_truth.keys())
    ground_truth_embeddings = embeddings_model.embed_documents(ground_truth_questions)
    similarities = cosine_similarity([user_query_embedding], ground_truth_embeddings).flatten()
    closest_idx = np.argmax(similarities)
    closest_similarity = similarities[closest_idx]
    if closest_similarity >= 0.5:
        return ground_truth_questions[closest_idx], closest_similarity
    else:
        return None, 0

# Store history of queries and responses
history = []

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/ask', methods=['POST'])
def ask():
    query = request.json.get('query').lower().strip()

    # Check if query is in predefined questions
    if query in predefined_qa:
        predefined_answer = predefined_qa[query]
        history.append((query, predefined_answer, None, None, 1.0))  # Save to history with perfect similarity score
        return jsonify({
            'status': 'answer',
            'answer': predefined_answer,
            'closest_question': None,
            'ground_truth_answer': None,
            'similarity_score': 1.0
        })

    if query.lower() == 'show graph':
        graph_path = generate_graph()  # Call the function to generate the graph
        return jsonify({
            'status': 'graph',
            'graph_path': graph_path,  # Return the path to the generated graph
            'similarity_scores': [item[4] for item in history],
            'queries': [item[0] for item in history]
        })

    try:
        model_response = bot(query)
        print(f"Model Response: {model_response}")

        # Extract the helpful answer without additional modifications
        predicted_answer = get_helpful_answer(model_response['result'])

    except Exception as e:
        print(f"Error in model response: {e}")
        return jsonify({'status': 'error', 'message': 'Failed to get a model response.'})

    closest_question, _ = find_closest_question(query, ground_truth)

    ground_truth_answer = None
    similarity_score = 0

    if closest_question is not None:
        ground_truth_answer = ground_truth[closest_question]

        # Calculate similarity score between helpful answer and ground truth answer
        predicted_embedding = embeddings_model.embed_query(predicted_answer)
        ground_truth_embedding = embeddings_model.embed_query(ground_truth_answer)

        # Calculate cosine similarity
        similarity_score = cosine_similarity([predicted_embedding], [ground_truth_embedding])[0][0]

    # Store the history (add similarity score)
    history.append((query, predicted_answer, closest_question, ground_truth_answer, similarity_score))

    return jsonify({
        'status': 'answer',
        'answer': predicted_answer,
        'closest_question': closest_question,
        'ground_truth_answer': ground_truth_answer,
        'similarity_score': similarity_score
    })

def generate_graph():
    queries = [item[0] for item in history]
    similarity_scores = [item[4] for item in history]
    plt.figure(figsize=(10, 6))

    # Create a bar chart
    for i in range(len(queries)):
        plt.bar(f"Q{i + 1}", similarity_scores[i] * 100, label=f"Q: {queries[i][:30]}...")

    plt.axhline(y=100, color='orange', linestyle='dashed', label='Ground Truth')
    plt.ylabel('Similarity Percentage (%)')
    plt.title('Similarity Scores of Questions')
    plt.legend()
    graph_path = 'static/similarity_graph.png'
    plt.savefig(graph_path)
    plt.close()  # Close the figure to prevent display
    return graph_path


# Fixed answers for different questions about Virat Kohli
fixed_answers = [
    "Virat Kohli is one of the best cricketers in the world. He was born on November 5, 1988, in Delhi, India. Kohli has captained the Indian national team in all formats. He is known for his aggressive batting style and consistency.",
    "Kohli is known for his excellent batting average and is often considered one of the greatest modern-day batsmen.",
    "He started his international career in August 2008 and quickly became a key player for India.",
    "Virat Kohli has won numerous awards, including the ICC ODI Player of the Year multiple times.",
    "He is married to Bollywood actress Anushka Sharma, and they have a daughter named Vamika.",
    "Kohli is also known for his fitness regime and promotes healthy living among his fans.",
    "He has led the Royal Challengers Bangalore (RCB) in the Indian Premier League (IPL) since 2013.",
    "Kohli has broken several records, including being the fastest player to score 8,000, 9,000, and 10,000 runs in ODIs.",
    "He is a prominent figure in philanthropy, supporting various causes through his charitable foundation.",
    "Virat Kohli has a massive fan following on social media, with millions of followers on platforms like Instagram and Twitter."
]


# Initialize lists to store questions and variable answers
questions = []
variable_answers = []


@app.route('/index1', methods=['GET', 'POST'])
def index1():
    results = []
    if request.method == 'POST':
        if request.form.get('action') == 'Submit Question':
            # Store question and answer
            question = request.form.get('question')
            answer = request.form.get('answer')

            # Append to lists
            questions.append(question)
            variable_answers.append(answer)

        elif request.form.get('action') == 'Show Scores':
            # Convert variable answers to TF-IDF vectors
            vectorizer = TfidfVectorizer()
            tfidf_variable = vectorizer.fit_transform(variable_answers).toarray()

            # Calculate scores for each variable answer
            for idx, variable_answer in enumerate(variable_answers):
                tfidf_fixed = vectorizer.transform(fixed_answers).toarray()

                # Calculate average scores across all fixed answers
                pearson_scores = []
                spearman_scores = []
                rmse_scores = []

                for fixed_vector in tfidf_fixed:
                    # Pearson correlation
                    pearson_corr, _ = pearsonr(fixed_vector, tfidf_variable[idx])
                    pearson_scores.append(pearson_corr)

                    # Spearman correlation
                    spearman_corr, _ = spearmanr(fixed_vector, tfidf_variable[idx])
                    spearman_scores.append(spearman_corr)

                    # RMSE
                    rmse = np.sqrt(mean_squared_error(fixed_vector, tfidf_variable[idx]))
                    rmse_scores.append(rmse)

                # Store the mean scores for display (along with the question for context)
                results.append({
                    'question': questions[idx],  # Just for display purposes
                    'pearson_corr': np.mean(pearson_scores),
                    'spearman_corr': np.mean(spearman_scores),
                    'rmse': np.mean(rmse_scores)
                })

    return render_template('index1.html', results=results)

#------------------------------------------------------------------------------------------------------

# Check if CUDA is available and set the device
device = "cuda" if torch.cuda.is_available() else "cpu"


# Configuration for model parameters and device settings
class CFG:
    device = device
    seed = 42
    generator = torch.Generator(device).manual_seed(seed)
    image_gen_steps = 20
    image_gen_model_id = "stabilityai/stable-diffusion-2"
    image_gen_guidance_scale = 9  # Image guidance scale
    image_gen_size = (512, 512)  # Image size


# Replace with your Hugging Face authentication token
auth_token = os.getenv('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')

# Load the Stable Diffusion model
image_gen_model = StableDiffusionPipeline.from_pretrained(
    CFG.image_gen_model_id,
    torch_dtype=torch.float16 if device == "cuda" else torch.float32,
    revision="fp16" if device == "cuda" else None
)
image_gen_model = image_gen_model.to(CFG.device)

# Function to generate an image from text
def generate_image_from_text(prompt):
    image = image_gen_model(
        prompt,
        num_inference_steps=CFG.image_gen_steps,
        generator=CFG.generator,
        guidance_scale=CFG.image_gen_guidance_scale
    ).images[0]
    return image

# Route to handle image generation from text
@app.route('/generate_image', methods=['POST'])
def generate_image_route():
    # Get the text input from the form
    text = request.form.get('text')

    # Generate image based on the text
    generated_image = generate_image_from_text(text)

    # Save the image to a BytesIO object to send it back as a file
    img_io = BytesIO()
    generated_image.save(img_io, 'PNG')
    img_io.seek(0)

    # Send the image as a response to be displayed on the HTML page
    return send_file(img_io, mimetype='image/png')
#--------------------------------------------------------------------------------------------------
# Directories for uploading images and saving graphs/cropped images
UPLOAD_FOLDER = 'static/uploads'
CROPPED_FOLDER = 'static/cropped_images'
GRAPH_FOLDER = 'static/graphs'
DATASET_FOLDER = 'static/dataset'

ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(CROPPED_FOLDER, exist_ok=True)
os.makedirs(GRAPH_FOLDER, exist_ok=True)

# Set the UPLOAD_FOLDER in app config
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# YOLO model for text detection
yolo_model = torch.hub.load('ultralytics/yolov5', 'yolov5s')

# EasyOCR reader for English and Hindi
reader = easyocr.Reader(['en', 'hi'])

# VGG16 model for image similarity
base_model = VGG16(weights='imagenet')
vgg_model = Model(inputs=base_model.input, outputs=base_model.get_layer('fc1').output)

# Utility function to check allowed file types
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

# Function to extract features using VGG16 model
def extract_features(img_path, model):
    img = image.load_img(img_path, target_size=(224, 224))
    img_data = image.img_to_array(img)
    img_data = np.expand_dims(img_data, axis=0)
    img_data = preprocess_input(img_data)
    features = model.predict(img_data)
    return features.flatten()

# Function to find similar images based on VGG16 feature extraction
def find_similar_images(input_image_path, dataset_dir=DATASET_FOLDER):
    dataset_image_paths = [
        os.path.join(dataset_dir, fname) for fname in os.listdir(dataset_dir)
        if fname.lower().endswith(('png', 'jpg', 'jpeg'))
    ]

    if not dataset_image_paths:
        return [], []

    input_image_features = extract_features(input_image_path, vgg_model)
    dataset_features = [extract_features(img_path, vgg_model) for img_path in dataset_image_paths]

    input_image_features = np.array(input_image_features).reshape(1, -1)
    dataset_features = np.array(dataset_features)
    similarities = cosine_similarity(input_image_features, dataset_features)

    similar_image_indices = np.argsort(similarities[0])[::-1]
    return [os.path.basename(dataset_image_paths[i]) for i in similar_image_indices[:5]], similarities.flatten()

# Function to generate similarity graph
def plot_similarity_graph(similarity_scores, input_image, similar_images):
    plt.figure(figsize=(10, 5))
    plt.plot([0] + list(range(1, len(similarity_scores) + 1)), [1] + list(similarity_scores), marker='o')
    plt.xticks(ticks=list(range(len(similar_images) + 1)), labels=[input_image] + similar_images, rotation=45)
    plt.title('Similarity Scores')
    plt.xlabel('Images')
    plt.ylabel('Similarity Score')
    plt.grid()

    graph_filename = f'similarity_graph_{datetime.datetime.now().strftime("%Y%m%d%H%M%S")}.png'
    graph_path = os.path.join(GRAPH_FOLDER, graph_filename)
    plt.savefig(graph_path)
    plt.close()

    return graph_filename



# Route for image-to-text extraction
@app.route('/image-text', methods=['GET', 'POST'])
def imagetext():
    extracted_text = ""
    uploaded_image = None

    if request.method == 'POST':
        if 'image' not in request.files:
            print("No file part in the request")
            return redirect(request.url)
        file = request.files['image']
        if file.filename == '':
            print("No file selected for uploading")
            return redirect(request.url)

        if file and allowed_file(file.filename):
            file_path = os.path.join(app.config['UPLOAD_FOLDER'], secure_filename(file.filename))
            file.save(file_path)
            uploaded_image = file.filename

            # Load the image using OpenCV
            image_cv = cv2.imread(file_path)

            # Check if the image is loaded correctly
            if image_cv is None:
                print("Error: Image not found or unable to read.")
                return redirect(request.url)

            # Detect text regions using YOLO
            results = yolo_model(file_path)
            detected_boxes = results.xyxy[0]

            # List to store extracted text
            extracted_text_list = []

            # Confidence threshold for text detection
            confidence_threshold = 0.5

            # Loop over detected boxes and extract text
            for i, box in enumerate(detected_boxes):
                x_min, y_min, x_max, y_max, confidence, class_idx = box

                if confidence > confidence_threshold:
                    x_min, y_min, x_max, y_max = map(int, [x_min, y_min, x_max, y_max])
                    cropped_img = image_cv[y_min:y_max, x_min:x_max]

                    cropped_path = os.path.join(CROPPED_FOLDER, f"cropped_{i}.jpg")
                    cv2.imwrite(cropped_path, cropped_img)

                    result = reader.readtext(cropped_img, detail=0)
                    if result:
                        extracted_text_list.append(" ".join(result))

            extracted_text = "\n".join(extracted_text_list)

            if not extracted_text_list:
                full_image_result = reader.readtext(file_path, detail=0)
                extracted_text = " ".join(full_image_result)

    return render_template('image-text.html', uploaded_image=uploaded_image, extracted_text=extracted_text)

# Route for similar image functionality
@app.route('/similar-image', methods=['GET', 'POST'])
def similarimage():
    if request.method == 'POST':
        file = request.files.get('image')
        if not file:
            return "No file provided", 400
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            img_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(img_path)

            similar_images, similarity_scores = find_similar_images(img_path)
            graph_image = plot_similarity_graph(similarity_scores[:5], filename, similar_images)

            return render_template('similar-image.html', uploaded_image=filename, similar_images=similar_images,
                                   graph_image=graph_image)
        else:
            return "Invalid file type", 400
    return render_template('similar-image.html')

# Route for showing generated similarity graph
@app.route('/graph')
def show_graph():
    graph_image = request.args.get('graph_image')
    graph_path = os.path.join(GRAPH_FOLDER, graph_image)
    if not os.path.exists(graph_path):
        return "Graph not found", 404
    return send_file(graph_path, mimetype='image/png')
#-------------------------------------------------------------------------------------------------------

if __name__ == "__main__":
    app.run(port=5004,debug=True)
