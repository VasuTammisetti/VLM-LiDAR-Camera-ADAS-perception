pipeline {
    agent any

    environment {
        DOCKER_IMAGE_TEST = 'vlm-adas-test'
        DOCKER_IMAGE_APP  = 'vlm-adas-app'
        REGISTRY          = 'tammisetti'
    }

    stages {
        stage('Checkout') {
            steps {
                checkout scm
            }
        }

        stage('Build Test Image') {
            steps {
                sh 'docker build -t ${DOCKER_IMAGE_TEST} -f docker/Dockerfile.test .'
            }
        }

        stage('Lint') {
            steps {
                sh 'docker run --rm ${DOCKER_IMAGE_TEST} flake8 src/ --max-line-length=120'
            }
        }

        stage('Unit Tests') {
            steps {
                sh '''
                    set +e
                    docker run --name vlm-adas-test-run \
                        ${DOCKER_IMAGE_TEST} \
                        pytest tests/ -v --tb=short --junitxml=/app/results.xml
                    EXIT_CODE=$?

                    mkdir -p test-results
                    docker cp vlm-adas-test-run:/app/results.xml test-results/results.xml || true
                    docker rm vlm-adas-test-run || true

                    exit $EXIT_CODE
                '''
            }
            post {
                always {
                    junit allowEmptyResults: true, testResults: 'test-results/results.xml'
                }
            }
        }

        stage('Build App Image') {
            when { branch 'main' }
            steps {
                sh 'docker build -t ${REGISTRY}/${DOCKER_IMAGE_APP}:${BUILD_NUMBER} -f docker/Dockerfile .'
                sh 'docker tag ${REGISTRY}/${DOCKER_IMAGE_APP}:${BUILD_NUMBER} ${REGISTRY}/${DOCKER_IMAGE_APP}:latest'
            }
        }

        stage('Push to Registry') {
            when { branch 'main' }
            steps {
                withCredentials([usernamePassword(
                    credentialsId: 'dockerhub-creds',
                    usernameVariable: 'DOCKER_USER',
                    passwordVariable: 'DOCKER_PASS'
                )]) {
                    sh 'echo $DOCKER_PASS | docker login -u $DOCKER_USER --password-stdin'
                    sh 'docker push ${REGISTRY}/${DOCKER_IMAGE_APP}:${BUILD_NUMBER}'
                    sh 'docker push ${REGISTRY}/${DOCKER_IMAGE_APP}:latest'
                }
            }
        }
    }

    post {
        always {
            sh 'docker system prune -f || true'
        }
        success { echo 'Pipeline passed!' }
        failure { echo 'Pipeline failed!' }
    }
}