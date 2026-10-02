from src.loaders import load_document


document = load_document("data/raw/sample.pdf")

print("CONTENT:")
print(document.content)


print("\nFILE TYPE:")
print(document.file_type)

print("\nMETADATA:")
print(document.metadata)