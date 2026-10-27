from app.speech_cleanup import clean_transcript

def test_removes_standalone_hesitations():
    assert clean_transcript("Emm, men tovarimni olib keldim") == "men tovarimni olib keldim"
    assert clean_transcript("Bu hmm juda yaxshi") == "Bu juda yaxshi"

def test_does_not_guess_or_damage_words():
    assert clean_transcript("Men tovarim desam avarim bo'lmasin") == "Men tovarim desam avarim bo'lmasin"
    assert clean_transcript("Emma bilan kelaman") == "Emma bilan kelaman"
    assert clean_transcript("") == ""
