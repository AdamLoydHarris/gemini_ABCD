import google.generativeai as genai
from ABCD import ABCD

# Configure the API Key

GOOGLE_API_KEY = 'your api key'
genai.configure(api_key=GOOGLE_API_KEY)
model = genai.GenerativeModel('gemini-pro')

### setup ABCD game
env = ABCD()
### set gemini to only respond Up, Down, Left, Right
### log moves to real -time track progress
### ask what it thinks is going on at the end of each task

if __name__ == "__main__":
    pass