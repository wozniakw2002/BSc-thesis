from django import forms

class ImageUploadForm(forms.Form):
    image = forms.ImageField()
    visualizations = forms.MultipleChoiceField(
        choices=[('gradcam', 'GradCam'), ('lime', 'LIME'), ('shap', 'SHAP')],
        widget=forms.CheckboxSelectMultiple,
        required=True
    )
