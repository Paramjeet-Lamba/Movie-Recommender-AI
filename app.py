"""
Project 4 — Movie Recommendation System (Standalone App)
===========================================================
Content-based recommender: Descriptions -> TF-IDF -> Cosine Similarity -> Top Matches
Now with filters: Industry, Year Range, Genre, Actor/Actress

Developed by Paramjeet Lamba

Run locally:
    pip install -r requirements.txt
    streamlit run app.py
"""

import pandas as pd
import streamlit as st
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

st.set_page_config(
    page_title="Movie Recommender AI | Paramjeet Lamba",
    page_icon="🎬",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ------------------------------------------------------------------
# 0. Global styling — dark theme with FIXED contrast for inputs/labels
# ------------------------------------------------------------------
CUSTOM_CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Poppins:wght@400;500;600;700;800&family=Inter:wght@400;500;600&display=swap');

html, body, [class*="css"] { font-family: 'Inter', sans-serif; }
#MainMenu, footer, header {visibility: hidden;}

.stApp {
    background: radial-gradient(circle at 20% 20%, #1f1147 0%, #0d0b1f 45%, #05040d 100%);
    color: #f3f1fa;
}

/* ---------- Title ---------- */
.app-title-wrap { text-align: center; padding: 10px 0 6px 0; }
.app-title {
    font-family: 'Poppins', sans-serif;
    font-weight: 800;
    font-size: 2.9rem;
    margin: 0;
    background: linear-gradient(90deg, #a78bfa, #f472b6, #60a5fa);
    background-size: 200% auto;
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    animation: shine 6s linear infinite;
}
.app-subtitle { color: #cfc9e4; font-size: 1rem; margin-top: 6px; }

/* ---------- Sidebar ---------- */
[data-testid="stSidebar"] {
    background: #140f28;
    border-right: 1px solid rgba(255,255,255,0.08);
}
[data-testid="stSidebar"] * { color: #f3f1fa !important; }

/* ---------- Cards ---------- */
.glass-card {
    background: rgba(255,255,255,0.045);
    border: 1px solid rgba(255,255,255,0.09);
    border-radius: 20px;
    padding: 22px;
}
.movie-card {
    background: rgba(255,255,255,0.045);
    border: 1px solid rgba(255,255,255,0.09);
    border-radius: 18px;
    padding: 18px 20px;
    margin-bottom: 12px;
    display: flex;
    justify-content: space-between;
    align-items: center;
    flex-wrap: wrap;
    gap: 8px;
}
.movie-card .rank {
    font-family: 'Poppins', sans-serif;
    font-weight: 800;
    font-size: 1.3rem;
    color: #a78bfa;
    margin-right: 14px;
}
.movie-card .title { font-weight: 600; font-size: 1.05rem; color: #fff; }
.movie-card .meta { color: #b9b2d4; font-size: 0.82rem; margin-top: 4px; margin-left: 2px; }
.movie-card .score-badge {
    background: linear-gradient(90deg, #8b5cf6, #ec4899);
    padding: 5px 14px;
    border-radius: 999px;
    font-size: 0.85rem;
    font-weight: 700;
    color: white;
    white-space: nowrap;
}
.tag {
    display: inline-block;
    background: rgba(167,139,250,0.16);
    border: 1px solid rgba(167,139,250,0.35);
    color: #d9d0ff !important;
    border-radius: 999px;
    padding: 2px 10px;
    font-size: 0.75rem;
    margin-right: 6px;
}

/* ---------- Buttons ---------- */
.stButton>button, .stDownloadButton>button {
    background: linear-gradient(90deg, #8b5cf6, #ec4899);
    color: white !important;
    border: none;
    border-radius: 999px;
    padding: 0.6em 1.7em;
    font-weight: 600;
    box-shadow: 0 8px 24px rgba(139, 92, 246, 0.35);
    transition: transform 0.15s ease, box-shadow 0.15s ease;
}
.stButton>button:hover, .stDownloadButton>button:hover {
    transform: translateY(-2px);
    box-shadow: 0 12px 28px rgba(236, 72, 153, 0.4);
}

/* ---------- Metrics ---------- */
[data-testid="stMetric"] {
    background: rgba(255,255,255,0.045);
    border: 1px solid rgba(255,255,255,0.09);
    border-radius: 16px;
    padding: 14px 10px;
}
[data-testid="stMetricLabel"] p { color: #cfc9e4 !important; }
[data-testid="stMetricValue"] { color: #fff !important; }

/* ---------- Widget labels (selectbox, slider, text input, etc.) ---------- */
[data-testid="stWidgetLabel"] p,
.stTextInput label p,
.stSlider label p,
.stSelectbox label p,
.stMultiSelect label p,
.stRadio label p {
    color: #efe9ff !important;
    font-weight: 600 !important;
}

/* ---------- Text input — fixed dark bg + white text ---------- */
.stTextInput input,
div[data-baseweb="base-input"] input,
div[data-baseweb="input"] input {
    background-color: #1c1730 !important;
    border: 1px solid rgba(255,255,255,0.18) !important;
    color: #ffffff !important;
    caret-color: #ffffff !important;
    border-radius: 12px !important;
    -webkit-text-fill-color: #ffffff !important;
}
.stTextInput input::placeholder { color: #9891b0 !important; opacity: 1 !important; }
.stTextInput input:focus {
    border: 1px solid #a78bfa !important;
    box-shadow: 0 0 0 1px #a78bfa !important;
}

/* ---------- Selectbox / Multiselect — fixed dark bg + white text ---------- */
div[data-baseweb="select"] > div {
    background-color: #1c1730 !important;
    border: 1px solid rgba(255,255,255,0.18) !important;
    border-radius: 12px !important;
    color: #ffffff !important;
}
div[data-baseweb="select"] * { color: #ffffff !important; }
div[data-baseweb="popover"] { background-color: #1c1730 !important; }
li[role="option"] {
    background-color: #1c1730 !important;
    color: #ffffff !important;
}
li[role="option"]:hover, li[aria-selected="true"] {
    background-color: #322a54 !important;
}
span[data-baseweb="tag"] { background-color: #6d28d9 !important; }

/* ---------- Sliders ---------- */
[data-testid="stSlider"] [data-baseweb="slider"] div { background: #a78bfa; }
div[data-testid="stSliderTickBarMin"], div[data-testid="stSliderTickBarMax"] { color: #a89fc4 !important; }

/* ---------- Radio ---------- */
.stRadio div[role="radiogroup"] label p { color: #efe9ff !important; }

/* ---------- Dataframe ---------- */
[data-testid="stDataFrame"] { border-radius: 14px; overflow: hidden; }

/* ---------- Expander ---------- */
.streamlit-expanderHeader p { color: #efe9ff !important; }

/* ---------- Footer ---------- */
.dev-footer {
    text-align: center;
    padding: 24px 0 6px 0;
    margin-top: 26px;
    border-top: 1px solid rgba(255,255,255,0.08);
    color: #8b84a3;
    font-size: 0.9rem;
}
.dev-footer span {
    background: linear-gradient(90deg, #a78bfa, #f472b6);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    font-weight: 700;
}

@keyframes shine { to { background-position: 200% center; } }

@media (max-width: 768px) {
    .app-title { font-size: 2rem; }
}
</style>
"""
st.markdown(CUSTOM_CSS, unsafe_allow_html=True)

# ------------------------------------------------------------------
# 1. Dataset — movies with industry, year, genre, cast + description
#    (Hollywood / Bollywood / Tollywood / Lollywood samples)
# ------------------------------------------------------------------
def get_movies_dataset() -> pd.DataFrame:
    data = [
        # title, industry, year, genre, cast, description

        # ---------------- Hollywood ----------------
        ("Interstellar", "Hollywood", 2014, "Sci-Fi", "Matthew McConaughey, Anne Hathaway",
         "space science fiction astronauts future adventure"),
        ("Inception", "Hollywood", 2010, "Sci-Fi", "Leonardo DiCaprio, Tom Hardy",
         "science fiction dreams technology thriller mind bending"),
        ("The Martian", "Hollywood", 2015, "Sci-Fi", "Matt Damon, Jessica Chastain",
         "space science fiction astronaut survival mars adventure"),
        ("Arrival", "Hollywood", 2016, "Sci-Fi", "Amy Adams, Jeremy Renner",
         "science fiction aliens language space mystery"),
        ("The Matrix", "Hollywood", 1999, "Sci-Fi", "Keanu Reeves, Carrie-Anne Moss",
         "science fiction technology artificial intelligence action"),
        ("Avatar", "Hollywood", 2009, "Sci-Fi", "Sam Worthington, Zoe Saldana",
         "science fiction space aliens adventure fantasy"),
        ("Blade Runner 2049", "Hollywood", 2017, "Sci-Fi", "Ryan Gosling, Harrison Ford",
         "science fiction dystopia detective future noir"),
        ("Dune", "Hollywood", 2021, "Sci-Fi", "Timothee Chalamet, Zendaya",
         "science fiction desert planet politics epic adventure"),
        ("Gravity", "Hollywood", 2013, "Sci-Fi", "Sandra Bullock, George Clooney",
         "space survival astronaut disaster thriller"),
        ("Ex Machina", "Hollywood", 2014, "Sci-Fi", "Alicia Vikander, Domhnall Gleeson",
         "artificial intelligence robot thriller psychological"),
        ("Titanic", "Hollywood", 1997, "Romance", "Leonardo DiCaprio, Kate Winslet",
         "romance drama ship ocean historical tragedy"),
        ("The Notebook", "Hollywood", 2004, "Romance", "Ryan Gosling, Rachel McAdams",
         "romance relationship love drama emotional"),
        ("La La Land", "Hollywood", 2016, "Romance", "Ryan Gosling, Emma Stone",
         "romance musical drama dreams jazz"),
        ("Pride and Prejudice", "Hollywood", 2005, "Romance", "Keira Knightley, Matthew Macfadyen",
         "romance period drama classic literature"),
        ("Silver Linings Playbook", "Hollywood", 2012, "Romance", "Bradley Cooper, Jennifer Lawrence",
         "romance comedy drama relationship recovery"),
        ("Avengers: Endgame", "Hollywood", 2019, "Action", "Robert Downey Jr., Chris Evans",
         "superhero action marvel time travel adventure"),
        ("Iron Man", "Hollywood", 2008, "Action", "Robert Downey Jr., Gwyneth Paltrow",
         "superhero action technology marvel engineering"),
        ("Spider-Man: No Way Home", "Hollywood", 2021, "Action", "Tom Holland, Zendaya",
         "superhero action marvel multiverse adventure"),
        ("The Dark Knight", "Hollywood", 2008, "Action", "Christian Bale, Heath Ledger",
         "superhero action crime batman thriller"),
        ("Mad Max: Fury Road", "Hollywood", 2015, "Action", "Tom Hardy, Charlize Theron",
         "action post apocalyptic chase survival desert"),
        ("John Wick", "Hollywood", 2014, "Action", "Keanu Reeves, Michael Nyqvist",
         "action revenge assassin thriller gun"),
        ("Mission: Impossible - Fallout", "Hollywood", 2018, "Action", "Tom Cruise, Henry Cavill",
         "action spy thriller stunt mission"),
        ("Gladiator", "Hollywood", 2000, "Action", "Russell Crowe, Joaquin Phoenix",
         "action historical revenge arena epic"),
        ("Jurassic Park", "Hollywood", 1993, "Adventure", "Sam Neill, Laura Dern",
         "dinosaurs science adventure action island"),
        ("Indiana Jones and the Last Crusade", "Hollywood", 1989, "Adventure", "Harrison Ford, Sean Connery",
         "adventure archaeology action treasure"),
        ("Get Out", "Hollywood", 2017, "Horror", "Daniel Kaluuya, Allison Williams",
         "horror thriller psychological social suspense"),
        ("A Quiet Place", "Hollywood", 2018, "Horror", "Emily Blunt, John Krasinski",
         "horror survival monster silence suspense"),
        ("Hereditary", "Hollywood", 2018, "Horror", "Toni Collette, Alex Wolff",
         "horror family supernatural psychological disturbing"),
        ("The Conjuring", "Hollywood", 2013, "Horror", "Vera Farmiga, Patrick Wilson",
         "horror haunted supernatural ghost family"),
        ("It", "Hollywood", 2017, "Horror", "Bill Skarsgard, Jaeden Martell",
         "horror clown supernatural childhood fear"),
        ("Superbad", "Hollywood", 2007, "Comedy", "Jonah Hill, Michael Cera",
         "comedy friendship teen coming of age funny"),
        ("The Hangover", "Hollywood", 2009, "Comedy", "Bradley Cooper, Zach Galifianakis",
         "comedy wedding chaos friendship funny"),
        ("Bridesmaids", "Hollywood", 2011, "Comedy", "Kristen Wiig, Maya Rudolph",
         "comedy wedding friendship funny women"),
        ("Deadpool", "Hollywood", 2016, "Comedy", "Ryan Reynolds, Morena Baccarin",
         "superhero comedy action irreverent violence"),
        ("Knives Out", "Hollywood", 2019, "Comedy", "Daniel Craig, Ana de Armas",
         "mystery comedy family murder detective"),
        ("The Shawshank Redemption", "Hollywood", 1994, "Drama", "Tim Robbins, Morgan Freeman",
         "drama prison friendship hope redemption"),
        ("Forrest Gump", "Hollywood", 1994, "Drama", "Tom Hanks, Robin Wright",
         "drama history life journey inspirational"),
        ("The Godfather", "Hollywood", 1972, "Drama", "Marlon Brando, Al Pacino",
         "crime drama family mafia power"),
        ("Whiplash", "Hollywood", 2014, "Drama", "Miles Teller, J.K. Simmons",
         "drama music mentor ambition intense"),
        ("Parasite", "Hollywood", 2019, "Drama", "Song Kang-ho, Choi Woo-shik",
         "drama class thriller family social satire"),
        ("Fight Club", "Hollywood", 1999, "Drama", "Brad Pitt, Edward Norton",
         "drama psychological identity anarchy underground"),
        ("Se7en", "Hollywood", 1995, "Thriller", "Brad Pitt, Morgan Freeman",
         "crime thriller serial killer detective dark"),
        ("Zodiac", "Hollywood", 2007, "Thriller", "Jake Gyllenhaal, Robert Downey Jr.",
         "crime thriller serial killer investigation mystery"),
        ("Gone Girl", "Hollywood", 2014, "Thriller", "Ben Affleck, Rosamund Pike",
         "thriller mystery marriage crime psychological"),
        ("Shutter Island", "Hollywood", 2010, "Thriller", "Leonardo DiCaprio, Mark Ruffalo",
         "thriller psychological mystery asylum twist"),
        ("Frozen", "Hollywood", 2013, "Animation", "Kristen Bell, Idina Menzel",
         "animation musical family sisters magic"),
        ("Toy Story", "Hollywood", 1995, "Animation", "Tom Hanks, Tim Allen",
         "animation family toys friendship adventure"),
        ("Spider-Man: Into the Spider-Verse", "Hollywood", 2018, "Animation", "Shameik Moore, Jake Johnson",
         "animation superhero multiverse action adventure"),
        ("The Lord of the Rings: The Fellowship of the Ring", "Hollywood", 2001, "Fantasy", "Elijah Wood, Ian McKellen",
         "fantasy epic quest adventure magic"),
        ("Harry Potter and the Sorcerer's Stone", "Hollywood", 2001, "Fantasy", "Daniel Radcliffe, Emma Watson",
         "fantasy magic school adventure wizard"),

        # ---------------- Bollywood ----------------
        ("Dilwale Dulhania Le Jayenge", "Bollywood", 1995, "Romance", "Shah Rukh Khan, Kajol",
         "romance love family drama tradition emotional"),
        ("Kabhi Khushi Kabhie Gham", "Bollywood", 2001, "Drama", "Shah Rukh Khan, Kajol",
         "family drama love relationship emotional tradition"),
        ("Kal Ho Naa Ho", "Bollywood", 2003, "Romance", "Shah Rukh Khan, Preity Zinta",
         "romance drama friendship emotional life"),
        ("Chennai Express", "Bollywood", 2013, "Comedy", "Shah Rukh Khan, Deepika Padukone",
         "comedy romance action journey train"),
        ("My Name Is Khan", "Bollywood", 2010, "Drama", "Shah Rukh Khan, Kajol",
         "drama identity journey inspirational social"),
        ("3 Idiots", "Bollywood", 2009, "Comedy", "Aamir Khan, R. Madhavan",
         "comedy friendship college education emotional drama"),
        ("Dangal", "Bollywood", 2016, "Drama", "Aamir Khan, Fatima Sana Shaikh",
         "sports wrestling drama family inspirational biography"),
        ("Taare Zameen Par", "Bollywood", 2007, "Drama", "Aamir Khan, Darsheel Safary",
         "drama childhood education learning emotional"),
        ("PK", "Bollywood", 2014, "Comedy", "Aamir Khan, Anushka Sharma",
         "comedy drama satire alien society"),
        ("Lagaan", "Bollywood", 2001, "Drama", "Aamir Khan, Gracy Singh",
         "sports cricket historical drama village"),
        ("Sultan", "Bollywood", 2016, "Drama", "Salman Khan, Anushka Sharma",
         "sports wrestling drama romance inspirational"),
        ("Bajrangi Bhaijaan", "Bollywood", 2015, "Drama", "Salman Khan, Kareena Kapoor",
         "drama emotional family journey inspirational border"),
        ("Tiger Zinda Hai", "Bollywood", 2017, "Action", "Salman Khan, Katrina Kaif",
         "action spy thriller mission rescue"),
        ("Dabangg", "Bollywood", 2010, "Action", "Salman Khan, Sonakshi Sinha",
         "action comedy police drama entertainer"),
        ("Bharat", "Bollywood", 2019, "Drama", "Salman Khan, Katrina Kaif",
         "drama family history journey emotional inspirational"),
        ("Padmaavat", "Bollywood", 2018, "Drama", "Ranveer Singh, Deepika Padukone",
         "historical drama romance war epic royalty"),
        ("Gully Boy", "Bollywood", 2019, "Drama", "Ranveer Singh, Alia Bhatt",
         "music drama rap struggle inspirational city"),
        ("Simmba", "Bollywood", 2018, "Action", "Ranveer Singh, Sara Ali Khan",
         "action comedy police crime drama"),
        ("Bajirao Mastani", "Bollywood", 2015, "Romance", "Ranveer Singh, Deepika Padukone",
         "historical romance drama war royalty"),
        ("83", "Bollywood", 2021, "Drama", "Ranveer Singh, Deepika Padukone",
         "sports cricket historical drama inspirational team"),
        ("Queen", "Bollywood", 2013, "Comedy", "Kangana Ranaut, Lisa Haydon",
         "comedy drama journey self discovery emotional"),
        ("Tanu Weds Manu", "Bollywood", 2011, "Romance", "Kangana Ranaut, R. Madhavan",
         "romance comedy wedding drama relationship"),
        ("Zindagi Na Milegi Dobara", "Bollywood", 2011, "Comedy", "Hrithik Roshan, Farhan Akhtar",
         "friendship comedy travel drama emotional adventure"),
        ("War", "Bollywood", 2019, "Action", "Hrithik Roshan, Tiger Shroff",
         "action spy thriller mission chase"),
        ("Kabhi Alvida Naa Kehna", "Bollywood", 2006, "Drama", "Shah Rukh Khan, Amitabh Bachchan",
         "drama relationship marriage family emotional"),
        ("Krrish", "Bollywood", 2006, "Action", "Hrithik Roshan, Priyanka Chopra",
         "superhero action fantasy adventure family"),
        ("Barfi!", "Bollywood", 2012, "Romance", "Ranbir Kapoor, Priyanka Chopra",
         "romance comedy drama disability emotional"),
        ("Rockstar", "Bollywood", 2011, "Romance", "Ranbir Kapoor, Nargis Fakhri",
         "romance music drama passion heartbreak"),
        ("Yeh Jawaani Hai Deewani", "Bollywood", 2013, "Romance", "Ranbir Kapoor, Deepika Padukone",
         "romance friendship travel drama youth"),
        ("Sanju", "Bollywood", 2018, "Drama", "Ranbir Kapoor, Vicky Kaushal",
         "drama biography addiction friendship redemption"),
        ("Andhadhun", "Bollywood", 2018, "Thriller", "Ayushmann Khurrana, Tabu",
         "thriller mystery crime dark comedy suspense"),
        ("Article 15", "Bollywood", 2019, "Drama", "Ayushmann Khurrana",
         "drama crime social justice investigation"),
        ("Vicky Donor", "Bollywood", 2012, "Comedy", "Ayushmann Khurrana, Yami Gautam",
         "comedy drama family relationship social"),
        ("Uri: The Surgical Strike", "Bollywood", 2019, "Action", "Vicky Kaushal, Yami Gautam",
         "action military patriotic mission war"),
        ("Raazi", "Bollywood", 2018, "Thriller", "Alia Bhatt, Vicky Kaushal",
         "thriller spy drama patriotic emotional"),
        ("Gangubai Kathiawadi", "Bollywood", 2022, "Drama", "Alia Bhatt",
         "drama biography crime empowerment struggle"),
        ("Highway", "Bollywood", 2014, "Drama", "Alia Bhatt, Randeep Hooda",
         "drama journey emotional freedom relationship"),
        ("Kahaani", "Bollywood", 2012, "Thriller", "Vidya Balan, Parambrata Chatterjee",
         "thriller mystery crime investigation suspense"),
        ("The Dirty Picture", "Bollywood", 2011, "Drama", "Vidya Balan",
         "drama biography glamour struggle industry"),
        ("Talvar", "Bollywood", 2015, "Thriller", "Irrfan Khan, Konkona Sen Sharma",
         "crime thriller mystery investigation drama"),
        ("Piku", "Bollywood", 2015, "Comedy", "Deepika Padukone, Amitabh Bachchan",
         "comedy drama family relationship journey"),
        ("Chak De! India", "Bollywood", 2007, "Drama", "Shah Rukh Khan",
         "sports hockey drama team inspirational"),
        ("Swades", "Bollywood", 2004, "Drama", "Shah Rukh Khan, Gayatri Joshi",
         "drama social village inspirational identity"),
        ("Rang De Basanti", "Bollywood", 2006, "Drama", "Aamir Khan, Soha Ali Khan",
         "drama youth patriotic revolution friendship"),
        ("Om Shanti Om", "Bollywood", 2007, "Romance", "Shah Rukh Khan, Deepika Padukone",
         "romance reincarnation drama industry revenge"),
        ("Housefull", "Bollywood", 2010, "Comedy", "Akshay Kumar, Deepika Padukone",
         "comedy chaos romance entertainer funny"),
        ("Baby", "Bollywood", 2015, "Action", "Akshay Kumar",
         "action spy thriller mission terrorism"),
        ("Airlift", "Bollywood", 2016, "Drama", "Akshay Kumar, Nimrat Kaur",
         "drama war rescue inspirational crisis"),
        ("Padman", "Bollywood", 2018, "Drama", "Akshay Kumar, Sonam Kapoor",
         "drama social innovation inspirational health"),
        ("Toilet: Ek Prem Katha", "Bollywood", 2017, "Comedy", "Akshay Kumar, Bhumi Pednekar",
         "comedy drama social marriage sanitation"),

        # ---------------- Tollywood ----------------
        ("Baahubali: The Beginning", "Tollywood", 2015, "Action", "Prabhas, Rana Daggubati",
         "action epic kingdom war fantasy adventure"),
        ("Baahubali 2: The Conclusion", "Tollywood", 2017, "Action", "Prabhas, Anushka Shetty",
         "action epic kingdom war fantasy revenge"),
        ("Arjun Reddy", "Tollywood", 2017, "Romance", "Vijay Deverakonda, Shalini Pandey",
         "romance drama love intense emotional college"),
        ("Pushpa: The Rise", "Tollywood", 2021, "Action", "Allu Arjun, Rashmika Mandanna",
         "action crime drama smuggling forest"),
        ("RRR", "Tollywood", 2022, "Action", "N. T. Rama Rao Jr., Ram Charan",
         "action historical friendship revolution epic drama"),
        ("Eega", "Tollywood", 2012, "Fantasy", "Nani, Samantha Ruth Prabhu",
         "fantasy revenge romance thriller unique"),
        ("Bharat Ane Nenu", "Tollywood", 2018, "Drama", "Mahesh Babu, Kiara Advani",
         "drama politics family inspirational leadership"),
        ("Magadheera", "Tollywood", 2009, "Action", "Ram Charan, Kajal Aggarwal",
         "action fantasy reincarnation romance epic"),
        ("Rangasthalam", "Tollywood", 2018, "Drama", "Ram Charan, Samantha Ruth Prabhu",
         "drama village politics revenge family"),
        ("Jersey", "Tollywood", 2019, "Drama", "Nani, Shraddha Srinath",
         "sports cricket drama family inspirational comeback"),
        ("Kirrak Party", "Tollywood", 2018, "Romance", "Nikhil Siddhartha, Simran Choudhary",
         "romance college drama friendship youth"),
        ("Ala Vaikunthapurramuloo", "Tollywood", 2020, "Action", "Allu Arjun, Pooja Hegde",
         "action family drama comedy entertainer"),
        ("Sarileru Neekevvaru", "Tollywood", 2020, "Action", "Mahesh Babu, Rashmika Mandanna",
         "action drama family comedy military"),
        ("Geetha Govindam", "Tollywood", 2018, "Romance", "Vijay Deverakonda, Rashmika Mandanna",
         "romance comedy drama misunderstanding college"),
        ("Mahanati", "Tollywood", 2018, "Drama", "Keerthy Suresh, Dulquer Salmaan",
         "drama biography cinema struggle inspirational"),
        ("Fidaa", "Tollywood", 2017, "Romance", "Varun Tej, Sai Pallavi",
         "romance village drama comedy family"),
        ("Ninnu Kori", "Tollywood", 2017, "Romance", "Nani, Nivetha Thomas",
         "romance drama emotional relationship"),
        ("Yevade Subramanyam", "Tollywood", 2015, "Drama", "Nani, Malavika Nair",
         "drama journey friendship adventure emotional"),
        ("Manam", "Tollywood", 2014, "Drama", "Nagarjuna, Naga Chaitanya",
         "drama family reincarnation emotional generations"),
        ("Uppena", "Tollywood", 2021, "Romance", "Panja Vaisshnav Tej, Krithi Shetty",
         "romance drama tragedy fishing village"),
        ("Sita Ramam", "Tollywood", 2022, "Romance", "Dulquer Salmaan, Mrunal Thakur",
         "romance war drama letters emotional"),
        ("Karthikeya 2", "Tollywood", 2022, "Mystery", "Nikhil Siddhartha, Anupama Parameswaran",
         "mystery adventure spiritual thriller temple"),
        ("HIT: The First Case", "Tollywood", 2020, "Thriller", "Vishwak Sen, Ruhani Sharma",
         "thriller crime investigation police mystery"),
        ("Jathi Ratnalu", "Tollywood", 2021, "Comedy", "Naveen Polishetty, Rahul Ramakrishna",
         "comedy friendship village funny drama"),
        ("Bheeshma", "Tollywood", 2020, "Comedy", "Nithiin, Rashmika Mandanna",
         "comedy romance drama family relationship"),
        ("Middle Class Abbayi", "Tollywood", 2017, "Romance", "Nithiin, Sarah-Jane Dias",
         "romance comedy drama youth family"),
        ("Krack", "Tollywood", 2021, "Action", "Ravi Teja, Shruti Haasan",
         "action drama police revenge crime"),
        ("Balagam", "Tollywood", 2023, "Drama", "Priyadarshi, Kavya Kalyanram",
         "drama family village comedy emotional"),
        ("Sye Raa Narasimha Reddy", "Tollywood", 2019, "Action", "Chiranjeevi, Nayanthara",
         "action historical revolution war epic"),
        ("Pokiri", "Tollywood", 2006, "Action", "Mahesh Babu, Ileana D'Cruz",
         "action crime police undercover thriller"),

        # ---------------- Lollywood ----------------
        ("Waar", "Lollywood", 2013, "Action", "Shaan Shahid, Meesha Shafi",
         "action military thriller terrorism patriotic"),
        ("Punjab Nahi Jaungi", "Lollywood", 2017, "Comedy", "Humayun Saeed, Mehwish Hayat",
         "comedy romance family drama relationship"),
        ("Parey Hut Love", "Lollywood", 2019, "Comedy", "Sheheryar Munawar, Maya Ali",
         "comedy romance family drama funny"),
        ("Laal Kabootar", "Lollywood", 2019, "Drama", "Ahmed Ali Akbar, Mansha Pasha",
         "crime drama thriller city mystery"),
        ("Khuda Kay Liye", "Lollywood", 2007, "Drama", "Shaan Shahid, Fawad Khan",
         "drama family religion society emotional identity"),
        ("Bol", "Lollywood", 2011, "Drama", "Humaima Malick, Atif Aslam",
         "drama family society gender emotional"),
        ("Ho Mann Jahaan", "Lollywood", 2015, "Drama", "Sheheryar Munawar, Mahira Khan",
         "drama friendship music youth emotional"),
        ("Jawani Phir Nahi Ani", "Lollywood", 2015, "Comedy", "Humayun Saeed, Hamza Ali Abbasi",
         "comedy friendship marriage chaos funny"),
        ("Na Maloom Afraad", "Lollywood", 2014, "Comedy", "Javed Sheikh, Fahad Mustafa",
         "comedy crime chaos funny heist"),
        ("Actor In Law", "Lollywood", 2016, "Comedy", "Fahad Mustafa, Mehwish Hayat",
         "comedy courtroom drama family funny"),
        ("Verna", "Lollywood", 2017, "Drama", "Mahira Khan",
         "drama social justice crime emotional"),
        ("Cake", "Lollywood", 2018, "Drama", "Aamina Sheikh, Sanam Saeed",
         "drama family relationship emotional siblings"),
        ("Superstar", "Lollywood", 2019, "Romance", "Bilal Ashraf, Mahira Khan",
         "romance drama cinema industry emotional"),
        ("Chalay Thay Saath", "Lollywood", 2017, "Drama", "Osman Khalid Butt, Syra Yousuf",
         "drama travel friendship emotional adventure"),
        ("Manto", "Lollywood", 2015, "Drama", "Sarmad Khoosat, Saba Qamar",
         "drama biography writer society history"),
        ("Moor", "Lollywood", 2015, "Drama", "Hameed Sheikh, Samiya Mumtaz",
         "drama family railway society emotional"),
        ("Load Wedding", "Lollywood", 2018, "Comedy", "Fahad Mustafa, Mehwish Hayat",
         "comedy drama family marriage social"),
        ("Yalghaar", "Lollywood", 2017, "Action", "Shaan Shahid, Ayesha Omar",
         "action military war patriotic mission"),
        ("Dobara Phir Se", "Lollywood", 2016, "Drama", "Adeel Hussain, Hareem Farooq",
         "drama romance relationship emotional"),
        ("Parwaaz Hai Junoon", "Lollywood", 2018, "Action", "Hamza Ali Abbasi, Hania Aamir",
         "action drama airforce patriotic military"),
    ]
    return pd.DataFrame(
        data,
        columns=["title", "industry", "year", "genre", "cast", "description"],
    )
# ------------------------------------------------------------------
# 2. Build the TF-IDF matrix + similarity matrix (cached)
#    Description is enriched with genre/cast/industry so filters and
#    similarity reinforce each other.
# ------------------------------------------------------------------
@st.cache_resource(show_spinner="Building the recommendation engine...")
def build_recommender():
    movies = get_movies_dataset()

    enriched_text = (
        movies["description"] + " "
        + movies["genre"].str.lower() + " "
        + movies["industry"].str.lower() + " "
        + movies["cast"].str.replace(",", " ", regex=False).str.lower()
    )

    vectorizer = TfidfVectorizer(stop_words="english")
    movie_matrix = vectorizer.fit_transform(enriched_text)

    similarity_matrix = cosine_similarity(movie_matrix)

    return {
        "movies": movies,
        "vectorizer": vectorizer,
        "movie_matrix": movie_matrix,
        "similarity_matrix": similarity_matrix,
    }


def recommend_movies(movie_title, movies, similarity_matrix, candidate_indices, number_of_recommendations=5):
    """Recommend movies similar to movie_title, restricted to candidate_indices (a filtered pool)."""
    if movie_title not in movies["title"].values:
        return pd.DataFrame(columns=["movie", "similarity"])

    movie_index = movies.index[movies["title"] == movie_title][0]
    allowed = set(candidate_indices) - {movie_index}

    similarity_scores = [(i, similarity_matrix[movie_index][i]) for i in allowed]
    similarity_scores = sorted(similarity_scores, key=lambda x: x[1], reverse=True)

    recommendations = []
    for index, score in similarity_scores[:number_of_recommendations]:
        row = movies.iloc[index]
        recommendations.append({
            "movie": row["title"],
            "industry": row["industry"],
            "year": row["year"],
            "genre": row["genre"],
            "cast": row["cast"],
            "similarity": round(float(score), 3),
        })

    return pd.DataFrame(recommendations)


def render_footer():
    st.markdown(
        """
        <div class="dev-footer">
            🎬 Movie Recommender AI &nbsp;•&nbsp; Built with Streamlit, TF-IDF &amp; Cosine Similarity<br>
            Developed by <span>Paramjeet Lamba</span>
        </div>
        """,
        unsafe_allow_html=True,
    )


# ------------------------------------------------------------------
# 3. Load data / engine
# ------------------------------------------------------------------
artifacts = build_recommender()
movies_df = artifacts["movies"]
similarity_matrix = artifacts["similarity_matrix"]

all_industries = sorted(movies_df["industry"].unique().tolist())
all_genres = sorted(movies_df["genre"].unique().tolist())
min_year, max_year = int(movies_df["year"].min()), int(movies_df["year"].max())

# Flatten individual actor names out of the comma-separated "cast" column
all_actors = sorted({
    name.strip()
    for cast_string in movies_df["cast"]
    for name in cast_string.split(",")
})

# ------------------------------------------------------------------
# 4. Sidebar — filters + recommendation controls
# ------------------------------------------------------------------
with st.sidebar:
    st.markdown("### 🎛️ Filters")

    selected_industries = st.multiselect(
        "Industry", options=all_industries, default=all_industries,
        help="Hollywood, Bollywood, Tollywood, Lollywood...",
    )

    selected_year_range = st.slider(
        "Year range", min_value=min_year, max_value=max_year,
        value=(min_year, max_year), step=1,
    )

    selected_genres = st.multiselect(
        "Genre", options=all_genres, default=all_genres,
        help="Comedy, Horror, Drama, Romance, Action...",
    )

    selected_actors = st.multiselect(
        "Hero / Heroine", options=all_actors, default=[],
        help="Leave empty to include all actors. Pick one or more to narrow the list.",
    )

    st.divider()
    st.markdown("### ⚙️ Recommender Settings")
    num_recs = st.slider(
        "Number of recommendations", min_value=1, max_value=10, value=5, step=1,
        help="How many similar movies to show.",
    )
    st.divider()
    st.caption("Method: Content-based filtering (TF-IDF + Cosine Similarity)")


def apply_filters(df: pd.DataFrame) -> pd.DataFrame:
    filtered = df[
        df["industry"].isin(selected_industries)
        & df["genre"].isin(selected_genres)
        & df["year"].between(selected_year_range[0], selected_year_range[1])
    ]
    if selected_actors:
        actor_mask = filtered["cast"].apply(
            lambda cast_string: any(actor in cast_string for actor in selected_actors)
        )
        filtered = filtered[actor_mask]
    return filtered


filtered_df = apply_filters(movies_df)

# ------------------------------------------------------------------
# 5. Title
# ------------------------------------------------------------------
st.markdown(
    """
    <div class="app-title-wrap">
        <h1 class="app-title">🎬 Movie Recommender AI</h1>
        <p class="app-subtitle">Descriptions → TF-IDF → Cosine Similarity → Top Matches</p>
    </div>
    """,
    unsafe_allow_html=True,
)

tab1, tab2, tab3 = st.tabs(["🎯 Get Recommendations", "📚 Movie Catalog", "🧠 How It Works"])

# ------------------------------------------------------------------
# 6. Tab 1 — Get Recommendations
# ------------------------------------------------------------------
with tab1:
    st.subheader("Pick a movie you liked")

    if filtered_df.empty:
        st.warning("No movies match your current filters. Try widening the Industry, Year, Genre, or Actor filters in the sidebar.")
    else:
        st.caption(f"{len(filtered_df)} movie(s) match your filters.")
        selected_movie = st.selectbox("Movie", options=filtered_df["title"].tolist())

        if st.button("🎬 Recommend Similar Movies", type="primary"):
            candidate_indices = filtered_df.index.tolist()
            results = recommend_movies(
                selected_movie, movies_df, similarity_matrix, candidate_indices, num_recs
            )

            if results.empty:
                st.warning("No recommendations found within the current filters. Try widening them.")
            else:
                st.markdown(f"**Because you liked _{selected_movie}_:**")
                for rank, row in enumerate(results.itertuples(), start=1):
                    st.markdown(
                        f"""
                        <div class="movie-card">
                            <div>
                                <span class="rank">#{rank}</span>
                                <span class="title">{row.movie}</span>
                                <div class="meta">
                                    <span class="tag">{row.industry}</span>
                                    <span class="tag">{row.year}</span>
                                    <span class="tag">{row.genre}</span>
                                    &nbsp;{row.cast}
                                </div>
                            </div>
                            <span class="score-badge">{row.similarity * 100:.1f}% match</span>
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )

    st.divider()
    st.subheader("Quick examples")
    st.caption("These always use the full catalog, ignoring the sidebar filters.")
    example_titles = ["Interstellar", "Sultan", "Baahubali: The Beginning"]
    cols = st.columns(len(example_titles))
    full_indices = movies_df.index.tolist()
    for col, title in zip(cols, example_titles):
        with col:
            st.write(f"**{title}** →")
            top_result = recommend_movies(title, movies_df, similarity_matrix, full_indices, 1)
            if not top_result.empty:
                st.write(
                    f"{top_result.iloc[0]['movie']} "
                    f"({top_result.iloc[0]['similarity'] * 100:.1f}% match)"
                )

# ------------------------------------------------------------------
# 7. Tab 2 — Movie Catalog
# ------------------------------------------------------------------
with tab2:
    st.subheader("Dataset overview")
    c1, c2, c3 = st.columns(3)
    c1.metric("Total movies", movies_df.shape[0])
    c2.metric("Matching current filters", filtered_df.shape[0])
    c3.metric("Vocabulary size (TF-IDF terms)", artifacts["movie_matrix"].shape[1])
    st.dataframe(
        filtered_df[["title", "industry", "year", "genre", "cast"]],
        use_container_width=True,
        hide_index=True,
    )

# ------------------------------------------------------------------
# 8. Tab 3 — How It Works
# ------------------------------------------------------------------
with tab3:
    st.markdown(
        """
1. **TF-IDF** converts each movie's description — enriched with its genre, industry,
   and cast — into a vector of numbers, one dimension per unique word across the
   whole dataset. A word that's common in one movie's text but rare across the
   others gets a higher weight.
2. **Cosine similarity** compares every movie's vector against every other
   movie's vector, producing a score between `0` (unrelated) and `1` (identical)
   based on the *angle* between the vectors — not their raw size.
3. **Filters** (Industry, Year range, Genre, Hero/Heroine) narrow down which
   movies you can pick from *and* which movies are eligible to be recommended —
   the similarity scores themselves are computed once across the full catalog,
   then only filtered-in candidates are shown.
4. To recommend movies similar to a chosen title, the app looks up that movie's
   row in the similarity matrix, sorts every other *eligible* movie by score,
   and returns the top matches (excluding the movie compared with itself, which
   is always `1.0`).
5. This is a **content-based** recommender — it only looks at text descriptions
   and metadata, not user behavior or ratings. Real-world systems usually combine
   this with **collaborative filtering** (what similar users liked) for better results.
"""
    )

render_footer()
