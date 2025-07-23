import os
import logging
from flask import Flask, request, jsonify
from langchain.prompts import PromptTemplate
from langchain_community.llms import OpenAI
from langchain.chains import LLMChain
import json
import re
import spacy
from rapidfuzz.fuzz import token_set_ratio
from langdetect import detect
from translatepy import Translator
import requests

app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024 


# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def load_visa_types(json_file_path):
    try:
        with open(json_file_path, 'r', encoding='utf-8') as file:
            return json.load(file)
    except (IOError, json.JSONDecodeError) as e:
        logger.error(f"Failed to load JSON data from {json_file_path}: {e}")
        return {}

def extract_visa_and_country(visa_recommendation_text):
    matches = re.findall(r'(\w+[\w\s]*) - ([\w\s]+)\s*Percentage Match:', visa_recommendation_text)
    return [(visa_type.strip(), country.strip()) for visa_type, country in matches]

def extract_percentages(visa_recommendation_text):
    percentage_pattern = re.compile(r'Percentage Match:\s*(\d+)%')
    percentages = percentage_pattern.findall(visa_recommendation_text)
    return [int(percentage) for percentage in percentages]

def extract_reasons(visa_recommendation_text):
    reason_pattern = re.compile(r'Reason:\s*(.*?)(?:\s*Suggestion:|$)', re.DOTALL)
    reasons = reason_pattern.findall(visa_recommendation_text)
    return [reason.strip() for reason in reasons]

def extract_suggestions(visa_recommendation_text):
    suggestion_pattern = re.compile(r'Suggestion:\s*(.*?)\s*Improved Percentage:', re.DOTALL)
    suggestions = suggestion_pattern.findall(visa_recommendation_text)
    return [suggestion.strip() for suggestion in suggestions]

def extract_percentages_after_improvement(visa_recommendation_text):
    improved_percentage_pattern = re.compile(r'Improved Percentage:\s*(\d+)%')
    improved_percentages = improved_percentage_pattern.findall(visa_recommendation_text)
    return [int(percentage) for percentage in improved_percentages]

def detect_and_translate(text):
    try:
        language = detect(text)
        if language != 'en':
            translator = Translator()
            result = translator.translate(text, destination_language="English")
            return result.result
        return text
    except Exception as e:
        logger.error(f"Translation failed: {e}")
        return text

def recommend_visa(qa_pairs):
    extracted_info = {}
    for key, value in list(qa_pairs.items()):
        if isinstance(value, list):
            for index, item in enumerate(value):
                if isinstance(item, dict):
                    for nested_key, nested_value in item.items():
                        new_key = f"{key}_{index}_{nested_key}"
                        extracted_info[new_key] = nested_value
                else:
                    new_key = f"{key}_{index}"
                    extracted_info[new_key] = item
            del qa_pairs[key]
    merged_input = {**qa_pairs, **extracted_info}
    qa_text = "".join(f"Question: {question}\nAnswer: {answer}\n" for question, answer in merged_input.items())

    translated_text = detect_and_translate(qa_text)

    prompt_template = f"""
    Based on the applicant profile provided in the Q&A section ({translated_text}),
    as an immigration law expert, recommend at least 32 visas that realistically match the profile and update, specifying the corresponding countries 
    such as the United States, Canada, United Kingdom, United Arab Emirates, Germany, and Australia.

    For each visa type identified as suitable, list it separately with each applicable country, even if the same visa type applies to multiple countries. 
    Determine the percentage match for each visa based on the profile and provide suggestions to improve this match. 
    Ensure the improved percentage is always realistically greater than the initial percentage match with a realistic margin.

    Format your response as follows:
    'Visa Name - Country 
    Percentage Match: Match greater than 10% - 
    Reason: (Reason of recommending that visa) - 
    Suggestion: (Suggestions to improve the match percentage) - Improved Percentage: X%'

    The improved percentage should vary based on the suggestions given, reflecting a realistic improvement rather than a uniform or maximum increase.
    """

    prompt = PromptTemplate(input_variables=["translated_text"], template=prompt_template)
    llm = OpenAI(temperature=0.0, max_tokens=2000)
    chain = LLMChain(llm=llm, prompt=prompt)
    result = chain.invoke(input={'translated_text': translated_text})
    return result

def match_visa_types(extracted_data, json_data, percentages, visa_recommendation_text, threshold=70):
    matching_data = []

    country_visa_dict = {}
    for entry in json_data:
        if isinstance(entry, dict) and 'visaType' in entry and 'countryName' in entry:
            country = entry['countryName'].lower()
            if country not in country_visa_dict:
                country_visa_dict[country] = []
            country_visa_dict[country].append(entry)

    for (visa_type, country), percentage in zip(extracted_data, percentages):
        country_lower = country.lower()
        visa_type_lower = visa_type.lower()
        
        # Match the country first
        matched_country = None
        max_country_similarity = 0
        for json_country in country_visa_dict.keys():
            similarity = token_set_ratio(country_lower, json_country)
            if similarity > max_country_similarity:
                max_country_similarity = similarity
                matched_country = json_country
                if max_country_similarity >= threshold:
                    break

        if matched_country:
            matched_entry = None
            max_visa_similarity = 0
            for entry in country_visa_dict[matched_country]:
                similarity = token_set_ratio(visa_type_lower, entry['visaType'].lower())
                if similarity > max_visa_similarity:
                    max_visa_similarity = similarity
                    matched_entry = entry
                    if max_visa_similarity >= threshold:
                        break

            if matched_entry:
                matched_entry = matched_entry.copy()
                matched_entry['percentage'] = percentage
                matching_data.append(matched_entry)
    return matching_data

@app.route('/recommend_visa/<id>', methods=['POST'])
def recommend_visa_endpoint(id):
    try:
        data = request.get_json()
        if not data:
            return jsonify({"error": "Invalid input data"}), 400

        openai_api_key = os.getenv("OPENAI_API_KEY")
        if not openai_api_key:
            return jsonify({"error": "Missing OpenAI API key"}), 500
        
        visa_recommendation_result = recommend_visa(data)
        visa_recommendation_text = visa_recommendation_result.get('text', 'No recommendation available')
        percentages = extract_percentages(visa_recommendation_text)
        extracted_data = extract_visa_and_country(visa_recommendation_text)
        reasons = extract_reasons(visa_recommendation_text)
        suggestions = extract_suggestions(visa_recommendation_text)
        improvements = extract_percentages_after_improvement(visa_recommendation_text)

        json_file_path = 'Visa_type_dataset.json'
        json_data = load_visa_types(json_file_path)

        matching_data = match_visa_types(extracted_data, json_data, percentages, visa_recommendation_text, threshold=70)
        
        structured_data = []
        for i, match in enumerate(matching_data):
            structured_entry = {
                "percentage": percentages[i],
                "reason": reasons[i] if i < len(reasons) else "No reason provided",
                "suggestion": suggestions[i] if i < len(suggestions) else "No suggestion provided",
                "percentage_after_improvement": improvements[i] if i < len(improvements) else "No improvement percentage provided"
            }

            match.pop('visaType - countryName', None)  
            structured_entry.update(match)
            structured_data.append(structured_entry)

        serialized_data = json.dumps(structured_data)
        size_of_data = len(serialized_data.encode('utf-8'))

        if size_of_data > 1000000:
            return jsonify({"error": "Data size is too large"}), 413

        endpoint_url = f"https://www.basepad.tech/api/immigrant/visaRecommendation/{id}"
        headers = {'Content-Type': 'application/json'}
        response = requests.post(endpoint_url, data=serialized_data, headers=headers)

        if response.status_code != 200:
            return jsonify({"error": "Failed to send data to endpoint"}), response.status_code

        return jsonify(structured_data), 200

    except Exception as e:
        logger.error(f"Error in recommend_visa_endpoint: {e}", exc_info=True)
        return jsonify({"error": "An error occurred"}), 500


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=False)
