import importlib.util

libs = ['pypdf', 'PyPDF2', 'pdfminer']
available = []

for lib in libs:
    if importlib.util.find_spec(lib):
        available.append(lib)

print(f"Available libraries: {available}")
