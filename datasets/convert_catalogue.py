import pandas as pd
import random
import json

INPUT_FILE = "/kaggle/input/amazon-product-data/dataset/train.csv" 
OUTPUT_TXT = "amazon_train_corpus.txt"

df = pd.read_csv(INPUT_FILE)

df = df[["TITLE", "BULLET_POINTS", "DESCRIPTION"]].copy()

df.reset_index(drop=True, inplace=True)

for col in ["TITLE", "BULLET_POINTS", "DESCRIPTION"]:
    if col not in df.columns:
        df[col] = ""
    else:
        df[col] = df[col].fillna("")

# ----------------------------
# TEMPLATES FOR VARIATION
# ----------------------------
TEMPLATES = [
    # 1. Simple direct
    "The product '{title}' comes with features like {bullet_points}. {description}",
    
    # 2. Conversational
    "Looking for something like '{title}'? It offers {bullet_points}. {description}",
    
    # 3. Feature-focused
    "Key highlights of '{title}' include {bullet_points}. {description}",
    
    # 4. Sales pitch
    "'{title}' is packed with features — {bullet_points}. {description}",
    
    # 5. Descriptive storytelling
    "Meet '{title}': {description} It also offers {bullet_points}.",
    
    # 6. Enthusiastic tone
    "You’ll love '{title}'! It has {bullet_points}. {description}",
    
    # 7. Problem-solving
    "If you need a solution with {bullet_points}, '{title}' is a great choice. {description}",
    
    # 8. Comparison-style
    "Among similar products, '{title}' stands out with {bullet_points}. {description}",
    
    # 9. Casual & friendly
    "'{title}' is here! Features include {bullet_points}. {description}",
    
    # 10. Clean & concise
    "'{title}': {bullet_points}. {description}",
    
    # 11. Professional tone
    "'{title}' offers {bullet_points} and is designed for users seeking {description}",
    
    # 12. Checklist tone
    "'{title}' — Features: {bullet_points}. Details: {description}",
    
    # 13. Marketing tagline style
    "'{title}' delivers {bullet_points}. Experience it yourself! {description}",
    
    # 14. Informative style
    "Product: '{title}'. Features: {bullet_points}. More info: {description}",
    
    # 15. Persuasive call-to-action
    "Choose '{title}' for {bullet_points}. {description}",
    
    # 16. Emphasis on specs
    "'{title}' specifications: {description} It also includes {bullet_points}.",
    
    # 17. Highlighting quality
    "'{title}' is known for {bullet_points}. {description} makes it a reliable option.",
    
    # 18. Short punchy version
    "'{title}' – {bullet_points}.",
    
    # 19. Informal vibe
    "Check out '{title}'! Comes with {bullet_points}. {description}",
    
    # 20. Story-style ending
    "'{title}' gives you {bullet_points}. {description} Perfect for your needs!"

     # 21. Review-style tone
    "'{title}' impresses users with {bullet_points}. {description}",
    
    # 22. Enthusiastic announcement
    "New arrival: '{title}' with {bullet_points}! {description}",
    
    # 23. Expert recommendation
    "Experts recommend '{title}' for its {bullet_points}. {description}",
    
    # 24. Focus on benefits
    "Enjoy benefits like {bullet_points} with '{title}'. {description}",
    
    # 25. Customer-centric
    "'{title}' is built for those who value {bullet_points}. {description}",
    
    # 26. Casual suggestion
    "Why not try '{title}'? It has {bullet_points}. {description}",
    
    # 27. Problem-solution narrative
    "Struggling to find a product with {bullet_points}? '{title}' is your answer. {description}",
    
    # 28. Lifestyle-oriented
    "'{title}' fits perfectly into your life with {bullet_points}. {description}",
    
    # 29. Social media friendly
    "'{title}' — trending now for its {bullet_points}. {description}",
    
    # 30. Focused on uniqueness
    "What makes '{title}' special? {bullet_points}. {description}",
    
    # 31. FAQ style
    "Looking for {bullet_points}? '{title}' might be what you need. {description}",
    
    # 32. Specs-first
    "With features like {bullet_points}, '{title}' is a standout choice. {description}",
    
    # 33. Purchase encouragement
    "Don’t miss '{title}' — featuring {bullet_points}. {description}",
    
    # 34. Trust-building tone
    "Many choose '{title}' because it offers {bullet_points}. {description}",
    
    # 35. Sleek marketing copy
    "Unleash the power of '{title}' — {bullet_points}. {description}",
    
    # 36. Bold claim
    "'{title}' sets a new standard with {bullet_points}. {description}",
    
    # 37. Minimal text (for short form)
    "'{title}' | {bullet_points} | {description}",
    
    # 38. Experience-focused
    "Experience {bullet_points} with '{title}'. {description}",
    
    # 39. Quality assurance style
    "'{title}' guarantees {bullet_points} so you get the best. {description}",
    
    # 40. Ending with a hook
    "'{title}' delivers {bullet_points}. {description} Ready to try it?"

    # 41. Informal & fun
    "Say hello to '{title}'! Packed with {bullet_points}. {description}",
    
    # 42. Question-style lead
    "Need a product that offers {bullet_points}? '{title}' is here. {description}",
    
    # 43. Emphasis on design
    "'{title}' boasts {bullet_points}, making it a stylish choice. {description}",
    
    # 44. Enthusiastic shopping pitch
    "Grab '{title}' now! Features include {bullet_points}. {description}",
    
    # 45. Highlighting performance
    "'{title}' performs exceptionally with {bullet_points}. {description}",
    
    # 46. Short & punchy
    "'{title}': {bullet_points}. {description}",
    
    # 47. Friendly advice
    "Looking for a great choice? '{title}' comes with {bullet_points}. {description}",
    
    # 48. User experience focus
    "Users will enjoy {bullet_points} with '{title}'. {description}",
    
    # 49. Storytelling narrative
    "Once you try '{title}', you'll love {bullet_points}. {description}",
    
    # 50. Simple description
    "'{title}' features {bullet_points}. {description}",
    
    # 51. Social proof style
    "People love '{title}' for {bullet_points}. {description}",
    
    # 52. Emphasis on quality
    "'{title}' ensures {bullet_points}. {description}",
    
    # 53. Casual promotion
    "Check out '{title}' — it offers {bullet_points}. {description}",
    
    # 54. Lifestyle angle
    "Make your life easier with '{title}', which comes with {bullet_points}. {description}",
    
    # 55. Direct call-to-action
    "Buy '{title}' now! Features: {bullet_points}. {description}",
    
    # 56. Informative tone
    "'{title}' provides {bullet_points}. More details: {description}",
    
    # 57. Attention-grabbing
    "Don’t miss out on '{title}'! {bullet_points}. {description}",
    
    # 58. Problem-solving
    "Struggling to find {bullet_points}? '{title}' solves it. {description}",
    
    # 59. Minimalist style
    "'{title}' — {bullet_points}",
    
    # 60. Customer-friendly
    "Customers will appreciate '{title}' with {bullet_points}. {description}",
    
    # 61. Highlighting innovation
    "'{title}' introduces {bullet_points}. {description}",
    
    # 62. Promotional tagline
    "'{title}': {bullet_points} — the perfect choice! {description}",
    
    # 63. Feature & detail
    "'{title}' comes with {bullet_points}. Specifications: {description}",
    
    # 64. Enthusiastic recommendation
    "You need '{title}'! It offers {bullet_points}. {description}",
    
    # 65. Persuasive narrative
    "Make the smart choice with '{title}', featuring {bullet_points}. {description}",
    
    # 66. Story-like lead
    "'{title}' brings {bullet_points} to your daily life. {description}",
    
    # 67. Tech/product focused
    "'{title}' — engineered for {bullet_points}. {description}",
    
    # 68. Lifestyle storytelling
    "Enhance your experience with '{title}' and {bullet_points}. {description}",
    
    # 69. Short sales pitch
    "'{title}' features {bullet_points}. Don’t wait! {description}",
    
    # 70. Highlight key points
    "Key features of '{title}' include {bullet_points}. {description}",
    
    # 71. Informal excitement
    "Wow! '{title}' offers {bullet_points}. {description}",
    
    # 72. Professional recommendation
    "'{title}' is recommended for its {bullet_points}. {description}",
    
    # 73. Shopping guide style
    "When shopping, consider '{title}' — it provides {bullet_points}. {description}",
    
    # 74. Descriptive and detailed
    "'{title}' includes {bullet_points}. {description}",
    
    # 75. Minimalist marketing
    "'{title}' | {bullet_points} | {description}",
    
    # 76. Attention grabber
    "Introducing '{title}'! {bullet_points}. {description}",
    
    # 77. Friendly suggestion
    "You might like '{title}' with {bullet_points}. {description}",
    
    # 78. Conversational question
    "Ever needed {bullet_points}? '{title}' has it! {description}",
    
    # 79. Engaging product story
    "'{title}' brings {bullet_points} to life. {description}",
    
    # 80. Informative snippet
    "Product '{title}' offers {bullet_points}. Details: {description}",
    
    # 81. Emphasis on benefits
    "'{title}' helps with {bullet_points}. {description}",
    
    # 82. Promotional narrative
    "Don’t miss '{title}'! Experience {bullet_points}. {description}",
    
    # 83. Descriptive headline
    "'{title}': Enjoy {bullet_points}. {description}",
    
    # 84. Positive experience
    "'{title}' guarantees {bullet_points}. {description}",
    
    # 85. Recommendation tone
    "We recommend '{title}' for {bullet_points}. {description}",
    
    # 86. Short & catchy
    "'{title}' — {bullet_points}.",
    
    # 87. Shopping excitement
    "Discover '{title}' with {bullet_points}. {description}",
    
    # 88. Highlight usage
    "Perfect for those needing {bullet_points}, '{title}' is ideal. {description}",
    
    # 89. Lifestyle benefit
    "'{title}' integrates {bullet_points} into your routine. {description}",
    
    # 90. Engaging headline
    "Here’s '{title}'! {bullet_points}. {description}",
    
    # 91. Feature first
    "{bullet_points} are included in '{title}'. {description}",
    
    # 92. Marketing flair
    "Step up your game with '{title}' — it offers {bullet_points}. {description}",
    
    # 93. Detailed yet casual
    "With {bullet_points}, '{title}' provides a complete solution. {description}",
    
    # 94. Focused on uniqueness
    "'{title}' stands out because of {bullet_points}. {description}",
    
    # 95. Friendly call-out
    "Hey! '{title}' has {bullet_points}. {description}",
    
    # 96. Engaging opening
    "Looking for {bullet_points}? Check out '{title}'. {description}",
    
    # 97. Tech-focused
    "'{title}' is equipped with {bullet_points}. {description}",
    
    # 98. Quality & reliability
    "'{title}' delivers {bullet_points}, ensuring great value. {description}",
    
    # 99. Casual recommendation
    "You’ll enjoy '{title}' with {bullet_points}. {description}",
    
    # 100. Hook ending
    "'{title}' comes with {bullet_points}. {description} Don’t miss out!"
]


def clean_bullets(bullets):
    """Safely clean bullet points into a single string."""
    # Convert to string first (avoid int/float errors)
    if bullets is None:
        return ""
    bullets = str(bullets)

    # Split on common delimiters (., •, -, \n)
    bullets_list = re.split(r"[•\-\n\.]", bullets)

    # Clean and deduplicate
    bullets_list = [b.strip() for b in bullets_list if b.strip()]
    bullets_list = list(dict.fromkeys(bullets_list))  # preserve order

    return " | ".join(bullets_list)

import re

def generate_corpus(df, templates, random_seed=42):
    """
    For every row in df, randomly select one template and fill it.
    Returns a fully randomized corpus.
    """
    random.seed(random_seed)
    corpus = []

    for _, row in df.iterrows():
        # Choose a random template for this row
        template = random.choice(templates)

        # Safely get values as strings
        title = str(row.get("TITLE", ""))
        description = str(row.get("DESCRIPTION", ""))
        bullet_points = row.get("BULLET_POINTS", "")

        # Clean bullet points
        bullets_cleaned = clean_bullets(bullet_points)

        # Fill the chosen template
        text = template.format(
            title=title,
            bullet_points=bullets_cleaned,
            description=description
        )

        corpus.append(text)

    # Final shuffle (optional, to mix rows even more)
    random.shuffle(corpus)

    return corpus

# Example usage
corpus = generate_corpus(df, TEMPLATES)


# Optional: Convert to single string
corpus_text = "\n".join(corpus)

# Save to a file
with open("amazon_train_corpus.txt", "w", encoding="utf-8") as f:
    f.write(corpus_text)

print("Corpus generated with", len(corpus), "rows")