import setuptools

with open("README.md", "r", encoding="utf-8") as f:
    long_description = f.read()


__version__ = "0.0.0"

REPO_NAME = "Image-Sharing-Plateform"
AUTHOR_USER_NAME = "atharva"
SRC_REPO = "image_sharing_plateform"
AUTHOR_EMAIL = "mishraatharva825@gmail.com"



setuptools.setup(
    name=SRC_REPO,
    version=__version__,
    author=AUTHOR_USER_NAME,
    author_email=AUTHOR_EMAIL,
    description="Project usses concept of Image Captioning to impliment this project",
    long_description=long_description,
    long_description_content="text/markdown",
    # url=f"https://github.com/mishraatharva/textsummarization.git",
    project_urls={
        "Bug Tracker": f"https://github.com/mishraatharva/textsummarization/issues",
    },
    package_dir={"": "src"},
    packages=setuptools.find_packages(where="src")
)