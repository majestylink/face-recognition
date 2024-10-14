import json

from django.http import JsonResponse
from django.shortcuts import render
from django.views import View

from django.utils.decorators import method_decorator
from django.views.decorators.csrf import csrf_exempt


from django.http import JsonResponse
from django.shortcuts import render
from django.views import View
import json

from server.utils.pytorch_util import process_image
from .models import Person


class IndexView(View):
    def get(self, request, *args, **kwargs):
        return render(request, 'index_1.html', {"title": "Majesty"})

    def post(self, request):
        # print(request.body)
        image = request.FILES.get('image', None)
        print(image, 'line 27')
        print("In view line 22")
        if image:
            print("In view line 30")
            result = process_image(image)
            if result == "Person not recognized":
                persons = [x.name for x in Person.objects.all()]
                return JsonResponse({'error': 'Person not recognized.', "persons": persons})
            person = Person.objects.get(label_name=result['predicted_label'])
            result = {
                'predicted_label': person.name,
                'probability_score': result['probability_score'],
                'predicted_image': request.build_absolute_uri(person.image.url),
                'date_of_birth': person.birth_year,
                'profession': person.occupation
            }

            return JsonResponse(result)
        else:
            print("In view line 52")
            return JsonResponse({'error': 'No image file provided.'})


    # return JsonResponse({'error': 'Unable to predict person.'})


        # Return the result as a JSON response
        # return JsonResponse(result_json, safe=False)
