# Azerbaijani Sign Language Recognition
> Striving for improved social inclusion

This projects is meant for creating a model for continuous transcription of the Azerbaijani Sign Language into Natural Language text.

## The Data

Video frames from 2 cameras set at different angles are collected. Each video corresponds to a specific sentence in Sign Language and is annotated by professional speakers using a service called Supervisely (available at *__https://supervise.ly/__*). Annotations include the starting and ending frames of a gloss (or a tag, the word in its Sign Language form)

## Installing / Getting started
> This section will be updated at a later stage

### Initial Configuration
> This section will be updated at a later stage

## Developing

To clone this project and build your awesome next-gen solution based on it:

```shell
git clone https://github.com/ADA-SITE-JML/sign-lang
```

### Building
> This section will be updated at a later stage

### Deploying / Publishing
> This section will be updated at a later stage

## Features

### dir_formatter.py
Automatically transforms the file directory format of the 2nd video stream to be the same as the manually structured file directory of the 1st video stream

### tags_to_sign_language.py
Converts the extracted JSON file from Supervisely into a DataFrame used by the model for training purposes

### temp_video_handler.py
We used a 3rd party web app called Supervisely (available at *__https://supervise.ly/__*) and this script enables us to set aside the yet-to-be uploaded videos and avoid having duplicate data in the annotation web app.

## Configuration
> This section will be updated at a later stage

## Contributing

"If you'd like to contribute, please fork the repository and use a feature
branch. Pull requests are warmly welcome."


## Licensing
> This section will be updated at a later stage
