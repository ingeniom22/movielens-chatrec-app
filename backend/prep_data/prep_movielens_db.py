import json
import pandas as pd
from faker import Faker
from faker.providers.person.en import Provider

usernames = list(set(Provider.first_names))
print(usernames)

fake = Faker()

users_df = pd.read_csv(
    "data/ml-100k/u.user",
    sep="|",
    names=["user_id", "age", "gender", "occupation", "zip_code"],
)

users_df["username"] = users_df["user_id"]
users_df["full_name"] = [fake.name() for _ in range(len(users_df))]
users_df["email"] = [fake.email() for _ in range(len(users_df))]
users_df["hashed_password"] = "$2b$12$EixZaYVK1fsbw1ZfbX3OXePaWxn96p36WQoeG6Lruj3vjPGga31lW"
users_df["disabled"] = False

users_df.set_index("username", inplace=True, drop=False)

print(users_df.head())
print(users_df.columns)

users_dict = users_df.to_dict(orient="index")

with open("app/router/mock_users_db.json", "w") as f:
    json.dump(users_dict, f, indent=4)
