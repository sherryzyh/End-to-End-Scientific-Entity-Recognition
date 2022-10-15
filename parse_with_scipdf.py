import scipdf

# article_dict = scipdf.parse_pdf_to_dict('example_data/futoma2017improved.pdf')  # return dictionary

# option to parse directly from URL to PDF, if as_list is set to True, output 'text' of parsed section will be in a list of paragraphs instead
article_dict = scipdf.parse_pdf_to_dict('https://www.biorxiv.org/content/biorxiv/early/2018/11/20/463760.full.pdf',
                                        as_list=False)
print(article_dict)