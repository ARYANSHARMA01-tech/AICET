import google.generativeai as genai
import os

def get_gemini_response(tree_name):
    """
    Gets information about a tree from the Gemini API, loading the key
    from environment variables.

    Args:
        tree_name: The name of the tree.

    Returns:
        A string containing the tree's description, advantages, and disadvantages,
        or an error message.
    """
    try:
        # Get the API key from the environment variable
        api_key = os.getenv("GEMINI_API_KEY")
        if not api_key:
            return "Error: GEMINI_API_KEY not found in the .env file. Please set it."

        # Configure the generative AI library with the API key
        genai.configure(api_key=api_key)

        # Initialize the Gemini 1.5 Flash model
        model = genai.GenerativeModel('gemini-1.5-flash')

        # Create a detailed prompt for the model
        prompt = f"""
        You are an expert botanist.
        Provide a detailed description of the tree species '{tree_name}'.
        After the description, list its main advantages and disadvantages in bullet points.
        Structure the output with the following markdown headers:
        
        ### Description
        
        ### Advantages
        
        ### Disadvantages
        """

        # Generate the content
        response = model.generate_content(prompt)
        return response.text
    except Exception as e:
        # Return a user-friendly error message
        return f"An error occurred: {e}. Please check your API key and network connection."
