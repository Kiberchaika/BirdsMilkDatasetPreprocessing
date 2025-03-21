import difflib
import html
from pathlib import Path

def get_diff(texts):
    """
    Create a multi-way diff showing matched and differing characters across all texts.
    
    Args:
        texts: List of text strings to compare
    
    Returns:
        List of lists, where each inner list contains tuples (char, texts_containing_char) for each text
    """
    # Validate input
    if len(texts) < 2:
        raise ValueError("At least 2 texts are needed for comparison")
    
    # Initialize result structure - for each text, create a list of (char, set of texts containing it)
    result = []
    
    for i, main_text in enumerate(texts):
        # For each character, track which texts contain it (starting with just the current text)
        char_presence = [set([i+1]) for _ in range(len(main_text))]
        
        # Compare against all other texts
        other_texts = [texts[j] for j in range(len(texts)) if j != i]
        other_indices = [j+1 for j in range(len(texts)) if j != i]  # 1-based indices for other texts
        
        for j, other_text in enumerate(other_texts):
            other_idx = other_indices[j]
            matcher = difflib.SequenceMatcher(None, main_text, other_text)
            
            # Look for matching blocks (sequences that match between the texts)
            for block in matcher.get_matching_blocks():
                if block.size > 0:
                    # For each matching character, add the other text's index to its presence set
                    for k in range(block.size):
                        idx_in_main = block.a + k
                        if idx_in_main < len(char_presence):  # Safety check
                            char_presence[idx_in_main].add(other_idx)
        
        # Convert presence sets to sorted strings (e.g., {1, 3} becomes "13")
        presence_strings = [''.join(map(str, sorted(presence))) for presence in char_presence]
        
        # Create list of (char, presence) tuples for this text
        text_result = list(zip(main_text, presence_strings))
        result.append(text_result)
    
    return result

def generate_html_from_diff(diff_result, filenames=None, output_file='text_comparison.html'):
    """
    Generate HTML visualization from diff result.
    
    Args:
        diff_result: Output from get_diff function
        filenames: Optional list of names for each text
        output_file: Output HTML file path. If None, returns HTML as string instead of writing to file
    
    Returns:
        String: HTML content as string if output_file is None
        None: Otherwise, saves the output to an HTML file
    """
    if filenames is None:
        filenames = [f"Text {i+1}" for i in range(len(diff_result))]
    
    css = """
    body { font-family: Arial, sans-serif; margin: 20px; max-width: 1200px; margin: 0 auto; }
    .comparison-grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(400px, 1fr)); gap: 20px; margin-top: 20px; }
    .comparison-cell { border: 1px solid #ddd; padding: 15px; border-radius: 5px; white-space: pre-wrap; word-break: break-word; }
    .comparison-header { font-weight: bold; margin-bottom: 10px; background-color: #f0f0f0; padding: 5px; border-radius: 3px; text-align: center; }
    .unique { background-color: #faa; }
    .in-all { background-color: #afa; }
    .shared { background-color: #aaf; }
    """
    
    html_parts = [
        "<!DOCTYPE html>",
        "<html>",
        "<head>",
        "<meta charset='utf-8'>",
        "<title>Text Comparison</title>",
        f"<style>{css}</style>",
        "</head>",
        "<body>",
        "<h1>Text Comparison</h1>"
    ]
    
    # Text comparison with highlights
    html_parts.append("<div class='comparison-grid'>")
    
    num_texts = len(diff_result)
    for i, text_result in enumerate(diff_result):
        html_parts.append(f"<div class='comparison-cell'>")
        html_parts.append(f"<div class='comparison-header'>{filenames[i]}</div>")
        
        current_class = ""
        current_text = ""
        html_content = []
        
        for char, presence in text_result:
            # Determine the class based on presence
            if len(presence) == 1:
                new_class = "unique"  # Unique to this text
            elif len(presence) == num_texts:
                new_class = "in-all"  # Present in all texts
            else:
                new_class = "shared"  # Shared with some but not all
            
            if new_class != current_class:
                if current_text:
                    html_content.append(f'<span class="{current_class}">{html.escape(current_text)}</span>')
                    current_text = ""
                current_class = new_class
            
            current_text += char
        
        # Add the last segment
        if current_text:
            html_content.append(f'<span class="{current_class}">{html.escape(current_text)}</span>')
        
        html_parts.append("".join(html_content))
        html_parts.append("</div>")
    
    html_parts.append("</div>")
    html_parts.append("</body></html>")
    
    # Join all HTML parts into a single string
    html_content = "\n".join(html_parts)
    
    # If output_file is None, return the HTML content as string
    if output_file is None:
        return html_content
    
    # Otherwise write to file
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write(html_content)
    print(f"Comparison saved to: {output_file}")

# Example usage
if __name__ == "__main__":
    example_texts = [
        "This is the first text for comparison. It contains some unique parts.",
        "This is the second text for comparison. It has differences from the first.",
        "This is the third text for comparison. It also has unique elements."
    ]
    
    diff_result = get_diff(example_texts)
    generate_html_from_diff(diff_result, output_file="comparison_result.html")