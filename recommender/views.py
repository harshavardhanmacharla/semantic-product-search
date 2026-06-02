from django.shortcuts import render
from .utils import smart_search

def home(request):
    results = []
    
    query = request.GET.get('query')

    if query:
        results = smart_search(query)

    return render(request, "index.html", {
        "results": results
    })