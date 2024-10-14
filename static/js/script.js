$(document).ready(function() {
    $('#upload-form').submit(function(event) {
        event.preventDefault(); // Prevent the form from submitting normally

        var formData = new FormData(this);

        // Get the CSRF token from the cookie
        var csrftoken = getCookie('csrftoken');

        // Set the CSRF token in the AJAX request headers
        $.ajaxSetup({
            beforeSend: function(xhr, settings) {
                if (!csrfSafeMethod(settings.type) && !this.crossDomain) {
                    xhr.setRequestHeader('X-CSRFToken', csrftoken);
                }
            }
        });

        $.ajax({
            url: '',
            type: 'POST',
            data: formData,
            processData: false,
            contentType: false,
            success: function(response) {
                console.log(response)
                var resultDiv = $('#result');
                var predictedImage = $('#predicted-image');
                var predictedLabel = $('#predicted-label');
                var probabilityScore = $('#probability-score');
                var dateOfBirth = $('#date-of-birth');
                var profession = $('#profession');

                // Update the predicted label and probability score
                predictedLabel.text('Predicted label: ' + response.predicted_label);
                probabilityScore.text('Probability score: ' + response.probability_score);
                dateOfBirth.text('Date of birth: ' + response.date_of_birth);
                profession.text('Profession: ' + response.profession);

                // Set the source of the predicted image
                predictedImage.attr('src', response.predicted_image);

                // Show the result section
                resultDiv.show();
            },
            error: function() {
                alert('Error occurred. Please try again.');
            }
        });
    });

    // Function to retrieve the CSRF token from the cookie
    function getCookie(name) {
        var cookieValue = null;
        if (document.cookie && document.cookie !== '') {
            var cookies = document.cookie.split(';');
            for (var i = 0; i < cookies.length; i++) {
                var cookie = cookies[i].trim();
                if (cookie.substring(0, name.length + 1) === (name + '=')) {
                    cookieValue = decodeURIComponent(cookie.substring(name.length + 1));
                    break;
                }
            }
        }
        return cookieValue;
    }

    // Function to check if the HTTP method is safe for CSRF protection
    function csrfSafeMethod(method) {
        return /^(GET|HEAD|OPTIONS|TRACE)$/.test(method);
    }
});
