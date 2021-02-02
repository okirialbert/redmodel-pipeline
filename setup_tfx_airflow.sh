#TFX and Airflow environment


PURPLE=$(tput setaf 125)
NORMAL=$(tput sgr0)

printf "${PURPLE}Installing Components Using pip${NORMAL}\n\n"

printf "${PURPLE}Installing Apache-Airflow${NORMAL}\n"
pip install apache-airflow 

printf "${PURPLE}Installing TFX${NORMAL}\n"
pip install tfx==0.26.0

printf "${PURPLE}Installing Neural Structured Learning Library${NORMAL}\n"
pip install neural_structured_learning

printf "${PURPLE}Installing docker server models${NORMAL}\n"
pip install docker

printf "\n${GREEN}Environment workshop installed${NORMAL}\n"
