# -*- coding: utf-8 -*-
"""Structured Recipes.ipynb

Automatically generated by Colab.

Original file is located at
    https://colab.research.google.com/drive/1l2Twm1wZs1GIevYexQiBrN9o0shKIamS
"""

!pip install mistralai
!pip install beautifulsoup4
!pip install urllib

import os, json, re
from mistralai.client import MistralClient
from mistralai.models.chat_completion import ChatMessage

from urllib.request import urlopen
from urllib.error import HTTPError
from bs4 import BeautifulSoup

api_key = "8cx2sY0Q0kNGjfOApJaK2oMjWOa5tGeP"
model = "mistral-small-latest"

client = MistralClient(api_key=api_key)

url = input("Enter the URL for the recipe you want structured: ")
try:
  webpage = urlopen(url).read().decode('utf-8')
  soup = BeautifulSoup(webpage, 'html.parser')
  prompt = soup.get_text()

  example = {
    "recipe_name": "Strawberry Shortcake Cookies",
    "author": ["Diana Moutsopoulos", "Amanda Holstein"],
    "tested_by": "Craig Ruff",
    "date_published": "June 8, 2024",
    "prep_time": "20 mins",
    "cook_time": "15 mins",
    "cool_time": "25 mins",
    "total_time": "1 hr",
    "servings": 14,
    "yield": 14,
    "ingredients": [
      {
        "name": "unsalted butter",
        "quantity": "1/2 cup",
        "for": "cookies"
      },
      {
        "name": "white sugar",
        "quantity": "3/4 cup",
        "for": "cookies"
      },
      {
        "name": "large egg",
        "quantity": "1",
        "for": "cookies",
        "notes": "at room temperature"
      },
      {
        "name": "vanilla extract",
        "quantity": "1 tablespoon",
        "for": "cookies"
      },
      {
        "name": "baking powder",
        "quantity": "1 teaspoon",
        "for": "cookies"
      },
      {
        "name": "baking soda",
        "quantity": "1/2 teaspoon",
        "for": "cookies"
      },
      {
        "name": "kosher salt",
        "quantity": "1/2 teaspoon",
        "for": "cookies"
      },
      {
        "name": "all-purpose flour",
        "quantity": "1 3/4 cups",
        "for": "cookies"
      },
      {
        "name": "heavy whipping cream",
        "quantity": "2 tablespoons",
        "for": "cookies"
      },
      {
        "name": "fresh strawberries",
        "quantity": "1 cup",
        "for": "cookies",
        "notes": "finely chopped, divided"
      },
      {
        "name": "unsalted butter",
        "quantity": "2 tablespoons",
        "for": "crumble"
      },
      {
        "name": "white sugar",
        "quantity": "1/4 cup",
        "for": "crumble"
      },
      {
        "name": "all-purpose flour",
        "quantity": "1/3 cup",
        "for": "crumble"
      }
    ],
    "instructions": [
      "Gather the ingredients.",
      "Preheat the oven to 350 degrees F (175 degrees C). Place racks in top 1/3 and bottom 1/3 positions. Line 2 large rimmed baking sheets with parchment paper; set aside.",
      "For cookies, beat butter and sugar with a stand mixer fitted with a paddle attachment on medium speed until light and creamy, about 2 minutes, stopping to scrape down sides of bowl as needed. Add egg and vanilla; beat until fully combined and smooth, about 30 seconds.",
      "Add baking powder, baking soda, salt, and flour to sugar mixture in mixer; beat until no dry streaks remain. Reduce mixer speed to low, and slowly add heavy cream; beat until fully combined, about 30 seconds.",
      "Gently fold 3/4 cup strawberries into cookie mixture with a rubber spatula until evenly distributed. Scoop dough into 14 equal portions (about 3 tablespoons each) onto baking sheets; arrange 7 cookies per baking sheet, spaced 1 1/2 inches apart.",
      "For crumble, stir butter, white sugar, and flour together in a small bowl until fully combined; using your fingers, rub mixture into small crumbles. Gently press crumble and remaining 1/4 cup strawberries evenly into tops of dough balls.",
      "Bake in the preheated oven until edges are golden brown, 12 to 15 minutes, rotating baking sheets between top and bottom racks and from front to back halfway through baking time. Cool on baking sheets for 5 minutes. Transfer cookies to a wire rack to cool completely, about 20 minutes."
    ]
  }

  chat_response = client.chat(
      model=model,
      messages=[ChatMessage(role="system", content="You are going to structure some text data into JSON. I will give you the text of a recipe, and you will output a structured JSON object including: ingredients, cooking time, quantities, etc. Use the same structure for all JSON objects you provide. Just return back the JSON object without any markdown. The resulting JSON object should be structured exactly like the provided example."),
                ChatMessage(role="user", content=f"Example: {example}"),
                ChatMessage(role="user", content=prompt)]
  )
  p = re.compile('(?<!\\\\)\'')
  s = p.sub('\"', chat_response.choices[0].message.content)

  with open("recipe.json", "w") as f:
      json.dump(json.loads(s), f)

  print("Recipe saved to recipe.json")
except(HTTPError):
  print("It looks like we aren't able to open that webpage. Please try another one.")