pipeline {
    agent any

    environment {
        DOCKER_IMAGE_TEST = 'vlm-adas-test'
        DOCKER_IMAGE_APP  = 'vlm-adas-app'
        REGISTRY          = 'vasutammisetti'
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
                    docker run --rm \
                        -v ${WORKSPACE}/test-results:/app/test-results \
                        ${DOCKER_IMAGE_TEST} \
                        pytest tests/ -v --tb=short --junitxml=test-results/results.xml
                '''
            }
            post {
                always {
                    junit 'test-results/results.xml'
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
