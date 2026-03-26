import os
import json
import base64
from PIL import Image
import pyttsx3
import io
import hashlib
from typing import TypedDict, Optional, List

import streamlit as st
import cloudinary
import cloudinary.uploader
import cloudinary.api

from langgraph.graph import StateGraph, END
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI

from dotenv import load_dotenv

load_dotenv()


# ── State schema required by modern LangGraph ──────────────────────────────────
class WorkflowState(TypedDict):
    image_path: str
    retrieved_docs: Optional[List]
    retrieved_embeddings: Optional[List[str]]
    retrieved_image_paths: Optional[List[str]]  # ✅ Added to hold retrieved image URLs/paths
    story: Optional[str]


# Step 1: Load and preprocess the VIST dataset
def preprocess_vist(json_file, image_dir):
    data = json.load(open(json_file))
    image_text_pairs = []
    for story in data["annotations"]:
        photo_ids = story["photo_ids"]
        sentences = story["story"]
        for photo_id, sentence in zip(photo_ids, sentences):
            image_path = os.path.join(image_dir, f"{photo_id}.jpg")
            if os.path.exists(image_path):
                image_text_pairs.append((image_path, sentence))
    return image_text_pairs


# Step 2: Store embeddings in Chroma
def store_embeddings_in_chroma(image_text_pairs, persist_directory):
    documents = [
        Document(page_content=text, metadata={"image_path": image_path})
        for image_path, text in image_text_pairs
    ]

    text_embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    vectorstore = Chroma.from_documents(
        documents=documents,
        embedding=text_embeddings,
        persist_directory=persist_directory,
    )
    vectorstore.persist()


# ── Helpers ────────────────────────────────────────────────────────────────────

def hash_image(image: Image.Image) -> str:
    buf = io.BytesIO()
    image.save(buf, format="PNG")
    return hashlib.md5(buf.getvalue()).hexdigest()


def speak_text(text: str):
    engine = pyttsx3.init()
    engine.say(text)
    engine.runAndWait()


def handle_speak(story):
    if story:
        speak_text(story[0]["content"])
    st.session_state.speak_triggered = False


def get_image_url(image_path: str) -> Optional[str]:
    cloudinary.config(
        cloud_name=os.getenv("CLOUDINARY_CLOUD_NAME"),
        api_key=os.getenv("CLOUDINARY_API_KEY"),
        api_secret=os.getenv("CLOUDINARY_API_SECRET"),
    )
    result = cloudinary.uploader.upload(image_path)
    url = result.get("secure_url")
    if url:
        print(f"Image uploaded: {url}")
    else:
        print("Upload failed.")
    return url


# ✅ Helper to convert local file paths to base64 so GPT-4o can read local VIST images
def get_image_uri(image_path: str) -> str:
    if image_path.startswith("http://") or image_path.startswith("https://"):
        return image_path  # Ready for API
    elif os.path.exists(image_path):
        with open(image_path, "rb") as image_file:
            encoded_string = base64.b64encode(image_file.read()).decode("utf-8")
            return f"data:image/jpeg;base64,{encoded_string}"
    return ""


# ── Agentic workflow ───────────────────────────────────────────────────────────

def agentic_workflow(image_url: str):
    workflow = StateGraph(WorkflowState)

    story_llm = ChatOpenAI(
        model="openai/gpt-4o",
        temperature=0.7,
        max_tokens=512,
        timeout=None,
        max_retries=2,
        api_key=os.getenv("LLM_API_KEY"),      
        base_url=os.getenv("LLM_BASE_URL"), 
    )

    text_embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )
    chroma_vectorstore = Chroma(
        persist_directory="chroma_store",
        embedding_function=text_embeddings,
    )

    def llm_node(state: WorkflowState) -> WorkflowState:
        print("LLM Node executed")
        return state

    def tool_node(state: WorkflowState) -> WorkflowState:
        print("Tool Node executed")
        
        caption_prompt = ChatPromptTemplate.from_messages([
            ("user", [
                {"type": "text", "text": "Describe the main subjects, setting, and mood of this image in one concise sentence."},
                {"type": "image_url", "image_url": {"url": "{image}"}}
            ])
        ])
        caption_chain = caption_prompt | story_llm
        caption_response = caption_chain.invoke({"image": state["image_path"]})
        dynamic_query = caption_response.content
        
        print(f"Dynamic query generated: {dynamic_query}")
        results = chroma_vectorstore.max_marginal_relevance_search(
            dynamic_query,
            k=2,
            fetch_k=20,
            lambda_mult=0.3
        )
        if results:
            state["retrieved_docs"] = results
        return state

    # ✅ Retriever Node: Now extracts the image paths from metadata
    def retriever_node(state: WorkflowState) -> WorkflowState:
        print("Retriever Node executed")
        state["retrieved_embeddings"] = [
            doc.page_content for doc in state["retrieved_docs"]
        ]
        state["retrieved_image_paths"] = [
            doc.metadata.get("image_path") for doc in state["retrieved_docs"]
        ]
        return state

    # ✅ Generator Node: Uses true Multi-Modal context by sending ALL images to GPT-4o
    def generator_node(state: WorkflowState) -> WorkflowState:
        print("Generator Node executed")

        if state.get("retrieved_embeddings"):
            texts = state["retrieved_embeddings"]
            image_paths = state.get("retrieved_image_paths", [])
            
            context_text = "\n".join([f"- {t}" for t in texts])
            
            # Construct complex multi-modal message
            content = [
                {
                    "type": "text", 
                    "text": (
                        "You are a creative storyteller. You have two sources of inspiration:\n"
                        "1. The user's uploaded image (the FIRST image below).\n"
                        f"2. Similar historical stories: \n{context_text}\n"
                        "3. The images associated with those historical stories (the SUBSEQUENT images below).\n\n"
                        "Create a captivating narrative that blends the context of the historical stories "
                        "with the specific visual details of the user's uploaded image."
                    )
                },
                {"type": "image_url", "image_url": {"url": state["image_path"]}} # User image
            ]
            
            # Append context images
            for img_path in image_paths:
                img_uri = get_image_uri(img_path)
                if img_uri:
                    content.append({"type": "image_url", "image_url": {"url": img_uri}})

            prompt = ChatPromptTemplate.from_messages([("user", content)])
            chain = prompt | story_llm
            response = chain.invoke({})

        else:
            content = [
                {
                    "type": "text",
                    "text": (
                        "Given the following image, use your imagination to craft an engaging "
                        "and immersive story. Describe the atmosphere, characters, and events "
                        "in a captivating way."
                    )
                },
                {"type": "image_url", "image_url": {"url": state["image_path"]}}
            ]
            prompt = ChatPromptTemplate.from_messages([("user", content)])
            chain = prompt | story_llm
            response = chain.invoke({})

        state["story"] = response.content
        return state

    # ✅ End Node: Renders the retrieved images for user verification
    def end_node(state: WorkflowState) -> WorkflowState:
        st.success("Story generated successfully!")
        
        # Display the contextual images found by Chroma
        if state.get("retrieved_image_paths"):
            st.write("### 🔍 Visual Inspiration Found in Database:")
            cols = st.columns(len(state["retrieved_image_paths"]))
            for idx, img_path in enumerate(state["retrieved_image_paths"]):
                with cols[idx]:
                    if img_path:
                        st.image(img_path, caption=f"Reference Image {idx+1}", use_container_width=True)

        st.session_state.speak_triggered = True
        st.text_area("Generated Story:", state["story"], height=200)
        st.session_state.story_history.append(
            {"role": "assistant", "content": state["story"]}
        )
        st.button(
            "Speak",
            on_click=handle_speak,
            args=(st.session_state.story_history,),
        )
        return state

    # ── Register nodes ─────────────────────────────────────────────────────────
    workflow.add_node("LLM", llm_node)
    workflow.add_node("Tools", tool_node)
    workflow.add_node("Retriever", retriever_node)
    workflow.add_node("Generator", generator_node)
    workflow.add_node("End", end_node)

    # ── Edges ──────────────────────────────────────────────────────────────────
    workflow.set_entry_point("LLM")
    workflow.add_edge("LLM", "Tools")

    workflow.add_conditional_edges(
        "Tools",
        lambda state: "Retriever" if state.get("retrieved_docs") else "Generator",
        {"Retriever": "Retriever", "Generator": "Generator"},
    )

    workflow.add_edge("Retriever", "Generator")
    workflow.add_edge("Generator", "End")
    workflow.add_edge("End", END)

    return workflow.compile()


# ── Streamlit app ──────────────────────────────────────────────────────────────

def run_streamlit_app():
    st.title("Visual Storytelling Generator")

    for key, default in [
        ("story_history", []),
        ("image_hash", None),
        ("speak_triggered", False),
    ]:
        if key not in st.session_state:
            st.session_state[key] = default

    uploaded_files = st.file_uploader(
        "Upload an image",
        accept_multiple_files=False,   
        type=["png", "jpg", "jpeg"],
    )

    if uploaded_files:
        image = Image.open(uploaded_files)
        st.image(image, caption="Uploaded Image", use_container_width=True)

        image.save("user_uploaded_photo.png")
        image_url = get_image_url("user_uploaded_photo.png")

        current_hash = hash_image(image)
        if st.session_state.image_hash is None:
            st.session_state.image_hash = current_hash
        elif st.session_state.image_hash != current_hash:
            st.session_state.story_history = []
            st.session_state.image_hash = current_hash
            st.warning("Image has changed — resetting story history.")

        with st.spinner("Running the Agentic AI Workflow…"):
            # ✅ Initialize new fields in the state
            initial_state: WorkflowState = {
                "image_path": image_url,
                "retrieved_docs": None,
                "retrieved_embeddings": None,
                "retrieved_image_paths": None, 
                "story": None,
            }
            app = agentic_workflow(image_url)
            app.invoke(initial_state)

    if not st.session_state.speak_triggered and st.session_state.story_history:
        st.text_area(
            "Generated Story:",
            st.session_state.story_history[0]["content"],
            height=200
        )
        st.button(
            "Speak",
            on_click=handle_speak,
            args=(st.session_state.story_history,),
        )


# ── Entry point ────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    vist_json = "path_to_vist_annotations.json"
    image_dir = "path_to_vist_images"
    persist_dir = "chroma_store"

    with st.spinner("Setting up embeddings…"):
        if os.path.exists(vist_json) and os.path.exists(image_dir):
            image_text_pairs = preprocess_vist(vist_json, image_dir)
        else:
            st.warning("Dataset not found → using demo data")
            image_text_pairs = [
                ("cloudinary_url_demo_image_1.jpg", "A group of friends enjoying a picnic in the park."),
                ("cloudinary_url_demo_image_2.jpg", "A family gathered around a dinner table sharing a meal."),
                ("cloudinary_url_demo_image_3.jpg", "A person hiking alone on a mountain trail during sunset."),
            ]

        store_embeddings_in_chroma(image_text_pairs, persist_dir)
        st.success("Embeddings ready!")

    run_streamlit_app()