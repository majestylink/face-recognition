import json

from django.http import JsonResponse
from django.shortcuts import render
from django.views import View

from django.utils.decorators import method_decorator
from django.views.decorators.csrf import csrf_exempt


class IndexView(View):
    # @method_decorator(csrf_exempt)
    # def dispatch(self, request, *args, **kwargs):
    #     return super().dispatch(request, *args, **kwargs)

    def get(self, request, *args, **kwargs):

        return render(request, 'index.html', {"title": "Majesty"})

    def post(self, request):
        print(request.FILES.get('image', None))
        # image = request.FILES['image']
        # print(image)
        result = {'name': 'Lionel Messi', 'age': 34, 'team': 'Paris Saint-Germain'}

        # Convert the result to a JSON string
        result_json = json.dumps(result)

        person_data = {
            'name': "predicted_person.name",
            'age': "predicted_person.age",
            'place_of_origin': "predicted_person.place_of_origin",
            'height': "predicted_person.height",
            'work_history': "predicted_person.work_history",
            'marital_status': "predicted_person.marital_status",
            'image_url': "predicted_person.image.url",
        }
        return JsonResponse(person_data)

    # return JsonResponse({'error': 'Unable to predict person.'})


        # Return the result as a JSON response
        # return JsonResponse(result_json, safe=False)
