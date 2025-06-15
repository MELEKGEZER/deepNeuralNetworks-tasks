import pandas as pd


themes = ["friendship", "hope", "sacrifice", "battle", "self development", "betrayal", "love", "dialogue"]


train_data = [
    ("I love spending time with my friends.", 0),  # friendship
    ("Hope is what keeps us going through tough times.", 1),  # hope
    ("Sometimes, sacrifice is necessary for the greater good.", 2),  # sacrifice
    ("The battle was fierce, but we overcame it.", 3),  # battle
    ("Personal growth is important for self development.", 4),  # self development
    ("Betrayal hurts, but it can teach us valuable lessons.", 5),  # betrayal
    ("Love is the strongest force in the universe.", 6),  # love
    ("Dialogue helps in resolving conflicts peacefully.", 7)  # dialogue
]

eval_data = [
    ("Friendship is built on trust and mutual understanding.", 0),  # friendship
    ("Without hope, there is no future.", 1),  # hope
    ("Sacrifice is a part of life, but it can be painful.", 2),  # sacrifice
    ("The battle for freedom was long and hard.", 3),  # battle
    ("To improve yourself, you must challenge yourself.", 4),  # self development
    ("Betrayal makes us question our relationships.", 5),  # betrayal
    ("Love is patient and kind.", 6),  # love
    ("Dialogue fosters understanding between people.", 7)  # dialogue
]


train_df = pd.DataFrame(train_data, columns=["text", "label"])
eval_df = pd.DataFrame(eval_data, columns=["text", "label"])

# CSV dosyalarına yazma
train_df.to_csv("train.csv", index=False)
eval_df.to_csv("eval.csv", index=False)

print("train.csv ve eval.csv dosyaları başarıyla oluşturuldu!")
