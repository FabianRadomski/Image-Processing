# Start from the  base file
FROM opencvcourses/opencv-docker:4.4.0

WORKDIR /home/imageprocessingcourse

# Import requirements
COPY requirements.txt .

# Install python (+ dependencies) and libgl (required by opencv)
RUN apt-get update -qq \
&& apt-get -y install python3-tk \
&& apt-get -y install libgl1-mesa-glx \
&& pip install -r requirements.txt

# Import project files
COPY . .

ENV PYTHONPATH="/home/imageprocessingcourse:${PYTHONPATH}"

# Run evaluator when starting the container
ENTRYPOINT ["sh", "/home/imageprocessingcourse/evaluator.sh"]
