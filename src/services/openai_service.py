import os
from openai import OpenAI
from src import config
from typing import Optional, List

# Initialize OpenAI client only if the key is available
client: Optional[OpenAI] = None
if config.OPENAI_API_KEY:
    client = OpenAI(api_key=config.OPENAI_API_KEY)
else:
    print("Warning: OpenAI API key not configured. Company descriptions will not be fetched.")

def get_company_summary(company_name: str) -> Optional[str]:
    """
    Fetches a one-paragraph summary of a company from OpenAI.

    Args:
        company_name: The name of the company.

    Returns:
        A string containing the company summary, or None if fetching fails
        or the API key is not configured.
    """
    if not client:
        return None

    try:
        prompt = f"Provide a concise, one-paragraph summary of what the company {company_name} does."

        response = client.chat.completions.create(
            model="gpt-3.5-turbo", # Or another suitable model
            messages=[
                {"role": "system", "content": "You are a helpful assistant that provides brief company summaries."},
                {"role": "user", "content": prompt}
            ],
            max_tokens=150, # Limit the length of the summary
            temperature=0.5, # Adjust creativity/factuality
            n=1,
            stop=None,
        )

        if response.choices:
            summary = response.choices[0].message.content.strip()
            return summary
        else:
            print(f"Warning: No summary received from OpenAI for {company_name}")
            return None

    except Exception as e:
        print(f"Error fetching company summary for {company_name} from OpenAI: {e}")
        return None


def get_economic_variables(company_name: str, symbol: str) -> Optional[List[str]]:
    """
    Fetches a list of key economic variables that typically influence a specific stock from OpenAI.

    Args:
        company_name: The name of the company.
        symbol: The stock symbol.

    Returns:
        A list of strings, where each string is an economic variable, or None if fetching fails
        or the API key is not configured.
    """
    if not client:
        return None

    try:
        prompt = f"List the top 3-5 key macroeconomic variables (like interest rates, inflation, GDP growth, unemployment rate, commodity prices, currency exchange rates, etc.) that are generally considered to significantly influence the stock price of {company_name} ({symbol}). Please provide the list as bullet points, with just the variable name."

        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "You are a financial analyst assistant providing lists of relevant macroeconomic variables for stocks."},
                {"role": "user", "content": prompt}
            ],
            max_tokens=100,
            temperature=0.3,
            n=1,
            stop=None,
        )

        if response.choices:
            content = response.choices[0].message.content.strip()
            # Parse the bulleted list
            variables = [line.strip().lstrip('-* ').strip() for line in content.split('\n') if line.strip().lstrip('-* ').strip()]
            if variables:
                return variables
            else:
                print(f"Warning: Could not parse variables from OpenAI response for {symbol}: {content}")
                return None
        else:
            print(f"Warning: No economic variables received from OpenAI for {symbol}")
            return None

    except Exception as e:
        print(f"Error fetching economic variables for {symbol} from OpenAI: {e}")
        return None



def get_economic_analysis(company_name: Optional[str], symbol: str, sector: Optional[str], industry: Optional[str]) -> Optional[str]:
    """
    Fetches a brief analysis from OpenAI about general economic conditions
    that might influence buying or trimming a specific stock.

    Args:
        company_name: The name of the company (optional).
        symbol: The stock symbol.
        sector: The company's sector (optional).
        industry: The company's industry (optional).

    Returns:
        A string containing the economic analysis, or None if fetching fails
        or the API key is not configured.
    """
    if not client:
        return None

    try:
        # Construct a more detailed prompt using available info
        prompt_parts = [
            f"Provide a brief (2-3 sentences) analysis on the general economic conditions under which an investor might consider buying or trimming the stock {symbol}"
        ]
        if company_name:
            prompt_parts.append(f" ({company_name})")
        prompt_parts.append(".")
        if sector:
            prompt_parts.append(f" The company operates in the {sector} sector")
            if industry:
                prompt_parts.append(f" within the {industry} industry.")
            else:
                prompt_parts.append(".")
        elif industry:
             prompt_parts.append(f" The company operates in the {industry} industry.")

        prompt_parts.append(" Focus on macroeconomic factors (like interest rate trends, inflation, sector-specific economic health, consumer spending, etc.) rather than company-specific news.")
        prompt = " ".join(prompt_parts)

        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "You are a financial analyst assistant providing concise economic context for stock investment decisions."},
                {"role": "user", "content": prompt}
            ],
            max_tokens=150, # Keep it brief
            temperature=0.6, # Allow for some nuanced interpretation
            n=1,
            stop=None,
        )

        if response.choices:
            analysis = response.choices[0].message.content.strip()
            return analysis
        else:
            print(f"Warning: No economic analysis received from OpenAI for {symbol}")
            return None

    except Exception as e:
        print(f"Error fetching economic analysis for {symbol} from OpenAI: {e}")
        return None


if __name__ == '__main__':
    # Example usage (for testing)
    test_company = "Microsoft"
    summary = get_company_summary(test_company)
    if summary:
        print(f"Summary for {test_company}:\n{summary}")
    else:
        print(f"Could not fetch summary for {test_company}.")

    test_company_no_key = "Apple"
    # Temporarily disable client for testing the 'no key' scenario
    original_client = client
    client = None
    summary_no_key = get_company_summary(test_company_no_key)
    if summary_no_key is None:
        print(f"Correctly handled missing API key for {test_company_no_key}.")
    else:
        print(f"Error: Unexpectedly got summary for {test_company_no_key} without API key.")
    client = original_client # Restore client