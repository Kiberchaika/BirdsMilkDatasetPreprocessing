import difflib

def three_way_diff(text1, text2, text3):
    """
    Create a three-way diff showing matches and differences between characters.
    Returns a list of tuples (char, texts_containing_char) for each text.
    """
    # Initialize character presence markers
    # Initially, each character is marked as unique to its text
    marks1 = ["1"] * len(text1)
    marks2 = ["2"] * len(text2)
    marks3 = ["3"] * len(text3)
    
    # Create SequenceMatchers for pairs
    sm12 = difflib.SequenceMatcher(None, text1, text2)
    sm13 = difflib.SequenceMatcher(None, text1, text3)
    sm23 = difflib.SequenceMatcher(None, text2, text3)
    
    # Process matching blocks between text1 and text2
    for match in sm12.get_matching_blocks():
        if match.size > 0:
            for i in range(match.size):
                a_idx = match.a + i
                b_idx = match.b + i
                
                # If character in text1 is also in text3, mark as in all three
                if "3" in marks1[a_idx]:
                    marks1[a_idx] = "123"
                    marks2[b_idx] = "123"
                # Otherwise mark as shared between text1 and text2
                else:
                    marks1[a_idx] = "12"
                    marks2[b_idx] = "12"
    
    # Process matching blocks between text1 and text3
    for match in sm13.get_matching_blocks():
        if match.size > 0:
            for i in range(match.size):
                a_idx = match.a + i
                c_idx = match.b + i
                
                # If character in text1 is also in text2, mark as in all three
                if "2" in marks1[a_idx]:
                    marks1[a_idx] = "123"
                    marks3[c_idx] = "123"
                # Otherwise mark as shared between text1 and text3
                else:
                    marks1[a_idx] = "13"
                    marks3[c_idx] = "13"
    
    # Process matching blocks between text2 and text3
    for match in sm23.get_matching_blocks():
        if match.size > 0:
            for i in range(match.size):
                b_idx = match.a + i
                c_idx = match.b + i
                
                # If character in text2 is also in text1, mark as in all three
                if "1" in marks2[b_idx]:
                    marks2[b_idx] = "123"
                    marks3[c_idx] = "123"
                # Otherwise mark as shared between text2 and text3
                else:
                    marks2[b_idx] = "23"
                    marks3[c_idx] = "23"
    
    # Fix any inconsistencies
    for marks, idx in [(marks1, 0), (marks2, 1), (marks3, 2)]:
        for i in range(len(marks)):
            # Sort the digits in each mark for consistency
            marks[i] = ''.join(sorted(marks[i]))
    
    return list(zip(text1, marks1)), list(zip(text2, marks2)), list(zip(text3, marks3))


def generate_html(marks1, marks2, marks3, name1, name2, name3):
    """Generate HTML output with highlighted differences in a three-column layout."""
    html_parts = [
        '<table style="width: 100%; border-collapse: collapse;">',
        '<tr>'
    ]
    
    def format_char(char, mark):
        """Helper function to format a single character with appropriate styling."""
        is_unique = mark != "123"
        
        if not is_unique:
            if char == '\n':
                return char + '<br>'
            elif char == ' ':
                return '&nbsp;'
            else:
                return char
        
        style_attr = ' style="background: #ffebee;"'
        
        if char == '\n':
            return f'<span{style_attr}>{char}</span><br>'
        elif char == ' ':
            return f'<span{style_attr}>&nbsp;</span>'
        else:
            return f'<span{style_attr}>{char}</span>'
    
    # Process each text block in columns
    for char_marks, name in [(marks1, name1), (marks2, name2), (marks3, name3)]:
        html_parts.extend([
            '<td style="vertical-align: top; padding: 10px; word-break: break-word; width: 33.3%;">',
            f"<h3>{name}</h3>",
            '<br/>',
            ''.join(format_char(char, mark) for char, mark in char_marks),
            '</td>'
        ])
    
    html_parts.extend([
        '</tr>',
        '</table>',
    ])
    
    return "\n".join(html_parts)


def main():
    # Example texts
    text1 = """This is the first text.
It has multiple lines.
This line is the same in all texts.
This line is different in text 1."""

    text2 = """This is the second text.
It has multiple lines.
This line is the same in all texts.
This line is different in text 2.
Text 2 has an extra line."""

    text3 = """This is the third text.
It also has multiple lines.
This line is the same in all texts.
This line is unique to text 3.
Text 3 also has an extra line."""

    # Get character-level differences
    marks1, marks2, marks3 = three_way_diff(text1, text2, text3)
    
    # Generate and save HTML output
    name1 = "Base"
    name2 = "Version 2"
    name3 = "Version 3"
    html_content = generate_html(marks1, marks2, marks3, name1, name2, name3)
    output_filename = "diff_result.html"
    with open(output_filename, "w") as f:
        f.write(html_content)
    print(f"\nHTML comparison saved to {output_filename}")

if __name__ == "__main__":
    main()