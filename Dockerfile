# docker build -t brennoli/project_2_application .
# docker push brennoli/project_2_application:latest

FROM openjdk:22-jdk-slim

# Copy Files
WORKDIR /usr/src/app
COPY . .

# Install
RUN ./mvnw -Dmaven.test.skip=true package

# Docker Run Command
EXPOSE 8082
CMD ["java","-jar","/usr/src/app/target/project_2-0.0.1-SNAPSHOT.jar"]