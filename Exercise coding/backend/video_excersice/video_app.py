# =====================================================================
# VIDEO ANALYSIS APP (Streamlit + Gemini)
# How to run (in the terminal, inside the backend folder, with (venv) on):
#     python -m streamlit run video_app.py
# The whole flow:
#   Choose source (Upload OR YouTube link) -> video shows -> pick a prompt
#   -> click Analyze
#      Upload:  video converted to base64 -> sent to Gemini as "data"
#      YouTube: the link is sent to Gemini as "file_uri" (no base64 needed)
#   -> answer shown on the page
# =====================================================================


# =====================================================================
# PART 1: IMPORTS - load the tools we need
# =====================================================================
import os, base64          # os = read settings (API key); base64 = turn video into text
import streamlit as st     # streamlit builds the web page; "st" is a short nickname
from dotenv import load_dotenv                              # reads your .env file
from langchain_google_genai import ChatGoogleGenerativeAI   # connects to Gemini
from langchain_core.prompts import ChatPromptTemplate       # builds the message to Gemini
from langchain_core.output_parsers import StrOutputParser   # turns Gemini's reply into plain text


# =====================================================================
# PART 2: CONNECT TO GEMINI
# =====================================================================
load_dotenv()   # opens .env and loads GEMINI_API_KEY

# llm = "Large Language Model" = the AI object we will talk to
llm = ChatGoogleGenerativeAI(
    model="gemini-3.8-flash",               # which Gemini version to use
    api_key=os.getenv("GEMINI_API_KEY")     # gets the key from .env (never type the key in code!)
)


# =====================================================================
# PART 3: THE PROMPT TEMPLATE - a "form with blanks"
#   "system" = tells Gemini WHO it is (its role)
#   "human"  = what the user sends: a question (text) + the video (media)
#   {curly braces} = blanks that get filled in later by chain.invoke()
# =====================================================================
prompt = ChatPromptTemplate.from_messages([
    ("system", "You are an expert video analyst. Analyze videos carefully and provide detailed descriptions."),
    ("human", [
        {
            "type": "text",
            "text": "{analysis_request}"    # blank 1: the question
        },
        {
            "type": "media",
            "data": "{video_data}",         # blank 2: the video as base64 text
            "mime_type": "{mime_type}"      # blank 3: file type, e.g. video/mp4
        }
    ])
])

# ---------------------------------------------------------------------
# PART 3B: A SECOND PROMPT FOR YOUTUBE LINKS
#   Same as above, but instead of "data" (base64 video) we send "file_uri"
#   (the YouTube link). Gemini opens the link and watches the video itself.
#   - no base64 needed
#   - no 20 MB limit
#   - the video must be PUBLIC
# ---------------------------------------------------------------------
prompt_youtube = ChatPromptTemplate.from_messages([
    ("system", "You are an expert video analyst. Analyze videos carefully and provide detailed descriptions."),
    ("human", [
        {
            "type": "text",
            "text": "{analysis_request}"    # blank 1: the question
        },
        {
            "type": "media",
            "file_uri": "{video_url}",      # blank 2: the YouTube link
            "mime_type": "video/mp4"        # fixed value - tells Gemini it's a video
        }
    ])
])


# =====================================================================
# PART 4: THE CHAINS - the "|" is a pipe: output of one step goes to the next
#   prompt (fill blanks) -> llm (Gemini answers) -> StrOutputParser (clean text)
# =====================================================================
chain = prompt | llm | StrOutputParser()                   # for uploaded videos
chain_youtube = prompt_youtube | llm | StrOutputParser()   # for YouTube links


# =====================================================================
# PART 5: PAGE TITLE + TWO COLUMNS
# =====================================================================
st.title("Video Analysis Using Gemini")   # big heading at the top

# st.radio = round buttons where you can pick ONE option
# whatever you pick is saved in "source"
source = st.radio("Video source", ["Upload a video", "YouTube link"], horizontal=True)

# start with "empty" values; they get filled in below depending on the source
uploaded_file = None
video_url = ""

col1, col2 = st.columns(2)   # split the page into LEFT (col1) and RIGHT (col2)


# =====================================================================
# PART 6: UPLOAD BOX or YOUTUBE LINK BOX (left side)
# =====================================================================
with col1:   # "with col1:" = put this on the LEFT side
    if source == "Upload a video":
        # shows an upload box; only mp4 files allowed
        # before anything is uploaded, uploaded_file is None (empty)
        uploaded_file = st.file_uploader("Upload your video", type=["mp4"])
    else:
        # shows a text box to paste the YouTube link
        # .strip() removes extra spaces at the start/end of the link
        video_url = st.text_input("Paste a YouTube link",
                                  placeholder="https://www.youtube.com/watch?v=...").strip()
        st.caption("The video must be public.")


# =====================================================================
# PART 7: SHOW THE VIDEO (right side)
# =====================================================================
with col2:   # put this on the RIGHT side
    if uploaded_file is not None:            # if a video has been uploaded...
        video_bytes = uploaded_file.read()   # get the raw video data (bytes)
        st.video(video_bytes)                # show the video player

        # len() = size in bytes; divide by 1024*1024 to get MB
        size_mb = len(video_bytes) / (1024 * 1024)

        # f-string puts the number inside the text; .1f = 1 decimal place (e.g. 43.4 MB)
        st.caption(f"Video size: {size_mb:.1f} MB")

        # Gemini rejects videos that are too big when sent this way
        if size_mb > 20:
            st.warning("This video is over 20 MB. Gemini may reject it - try a smaller video.")
    elif video_url:                          # if a YouTube link was pasted...
        st.video(video_url)                  # st.video can play YouTube links too!
    else:                                    # no video yet
        st.write("Welcome to the Video Analysis App!")


# =====================================================================
# PART 8: CHOOSE A PROMPT (Exercise 1 from slide 8)
# =====================================================================
# a list of prompts to choose from
exercise_prompts = [
    "Describe what's happening in this video.",
    "Give a detailed summary of this video.",
    "Summarize this video in a few short bullets.",
    "Generate a paragraph that summarizes this video, with corresponding timecodes.",
    "Summarize this video. Then create a quiz with an answer key based on the information in this video.",
    "Write my own question...",
]

# st.selectbox = dropdown menu; whatever you pick is saved in "choice"
choice = st.selectbox("Choose a prompt (Exercise 1)", exercise_prompts)

if choice == "Write my own question...":
    question = st.text_input("Enter a question")   # show a text box to type your own
else:
    question = choice                              # use the prompt picked from the dropdown


# =====================================================================
# PART 9: THE ANALYZE BUTTON - check for mistakes first
# =====================================================================
if st.button("Analyze video"):          # True only at the moment the button is clicked
    if uploaded_file is None and not video_url:   # no video AND no link -> red error message
        st.error("Please upload a video or paste a YouTube link first.")
    elif not question:                  # no question -> red error message
        st.error("Please enter a question.")
    else:                               # everything OK -> do the analysis

        # =============================================================
        # PART 10: SEND TO GEMINI AND SHOW THE RESULT
        # =============================================================
        # spinner = loading animation while we wait for Gemini
        with st.spinner("Analyzing video... this can take up to a minute"):
            try:   # try = attempt this; if it fails, jump to "except"
                if uploaded_file is not None:
                    # ---- UPLOADED VIDEO ----
                    # convert the video into base64 text (same as slide 5)
                    encoded_video = base64.b64encode(video_bytes).decode()

                    # fill in the blanks from PART 3 and send everything to Gemini
                    res = chain.invoke({
                        "analysis_request": question,          # -> {analysis_request}
                        "video_data": encoded_video,           # -> {video_data}
                        "mime_type": uploaded_file.type        # -> {mime_type}, e.g. video/mp4
                    })
                else:
                    # ---- YOUTUBE LINK ----
                    # fill in the blanks from PART 3B - just the question and the link
                    res = chain_youtube.invoke({
                        "analysis_request": question,          # -> {analysis_request}
                        "video_url": video_url                 # -> {video_url}
                    })

                st.subheader("Result")   # small heading
                st.write(res)            # show Gemini's answer on the page
            except Exception as e:
                # if anything goes wrong (e.g. video too big), show a red message instead of crashing
                st.error(f"Something went wrong: {e}")
