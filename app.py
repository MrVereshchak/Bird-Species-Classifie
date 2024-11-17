# Bird Species Classifier
# ========================
# This script uses a FastAI model to classify bird species from Ohio and provides a user-friendly interface using Gradio.

# Imports
from fastai.vision.all import *  # FastAI library for computer vision tasks
import gradio as gr  # Gradio library for creating web-based UIs

# Load Pre-trained Model
learn = load_learner('bird_model.pkl')  # Load the trained bird species classifier

# Bird Categories
categories = [
    "Northern Cardinal", "Blue Jay", "American Crow", "Black-capped Chickadee",
    "White-breasted Nuthatch", "Mourning Dove", "House Sparrow", "House Finch",
    "Downy Woodpecker", "Red-bellied Woodpecker", "Tufted Titmouse", "Carolina Wren",
    "Eastern Screech-Owl", "Great Horned Owl", "American Robin", "Barn Swallow",
    "Tree Swallow", "Baltimore Oriole", "Eastern Bluebird", "Ruby-throated Hummingbird",
    "Red-winged Blackbird", "Common Yellowthroat", "Indigo Bunting", "Eastern Meadowlark",
    "Canada Goose", "Snow Goose", "Sandhill Crane", "Yellow Warbler", "Warbling Vireo",
    "Swainson's Thrush", "Northern Parula", "Black-throated Green Warbler", "Hermit Thrush",
    "Rose-breasted Grosbeak", "Dark-eyed Junco", "American Tree Sparrow", "Snow Bunting",
    "Rough-legged Hawk", "Common Redpoll", "Pine Siskin", "Mallard", "Wood Duck",
    "Great Blue Heron", "Killdeer", "Belted Kingfisher", "Spotted Sandpiper",
    "Double-crested Cormorant", "Bald Eagle", "Red-tailed Hawk", "Cooper's Hawk",
    "Sharp-shinned Hawk", "Peregrine Falcon", "American Kestrel", "Osprey",
    "Northern Flicker", "Pileated Woodpecker", "Cedar Waxwing", "Yellow-bellied Sapsucker",
    "Scarlet Tanager", "Northern Mockingbird", "Brown Thrasher", "Orchard Oriole",
    "Eastern Phoebe", "Great Crested Flycatcher"
]

# Ensure categories is already sorted in the same way as learn.dls.vocab
categories = sorted(learn.dls.vocab)

# Image Classification Function
def classify_image(img):
    """
    Classify an input image to determine the bird species.

    Args:
        img (PIL Image): The input bird image.

    Returns:
        dict: A dictionary of bird species and their corresponding probabilities.
    """
    pred, idx, probs = learn.predict(img)
    return dict(zip(sorted(categories), map(float, probs)))

# Gradio Interface
intf = gr.Interface(
    fn=classify_image,  # Classification function
    inputs=gr.Image(type='pil'),  # Input type: image
    outputs=gr.Label(),  # Output type: label with probabilities
    examples=[  
        'Birds Examples/Common Yellowthroat 1.jpg', 
        'Birds Examples/Common Yellowthroat 2.jpg', 
        'Birds Examples/Mourning Dove.jpg', 
        'Birds Examples/Killdeer.jpg', 
        'Birds Examples/Sharp-shinned Hawk.jpg',
        'Birds Examples/Ruby-throated Hummingbird 1.jpg', 
        'Birds Examples/Ruby-throated Hummingbird 2.jpg'
    ],
    title="Bird Species Classifier",
    description="Upload a bird image to classify its species. This model was trained on bird species from Ohio.",
)

# Launch Gradio Interface
if __name__ == "__main__":
    intf.launch(inline=False)
