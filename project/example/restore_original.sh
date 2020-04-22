if [[ -f ../spiders/Dockerfile.BAK ]]
then
    mv ../spiders/Dockerfile.BAK ../spiders/Dockerfile
    echo 'Original spiders Dockerfile restored'
else
    echo 'Original spiders Dockerfile already restored'
fi

if [[ -f ../prediction/Dockerfile.BAK ]]
then
    mv ../prediction/Dockerfile.BAK ../prediction/Dockerfile
    echo 'Original prediction Dockerfile restored'
else
    echo 'Original prediction Dockerfile already restored'
fi