#!/bin/bash

# Create empty line at the end if doesn't exist
sed -i '$a\' users.txt

# declare empty array
ENTRIES=()

while read line; do
    if [[ ! -z "$line" ]]
    then
        ENTRIES+=("$line")
    fi
done < users.txt

GROUP_INFO=($(echo ${ENTRIES[0]} | tr "," " ")) 
GROUPNAME=${GROUP_INFO[0]}
GROUPID=${GROUP_INFO[1]}

addgroup --gid ${GROUP_INFO[1]} ${GROUP_INFO[0]}

echo ${GROUPNAME}

for user in ${ENTRIES[@]:1}
do
    USER_INFO=($(echo $user | tr "," " "))
    useradd -l -u ${USER_INFO[1]} -g ${USER_INFO[1]} ${USER_INFO[0]}
    usermod -a -G ${GROUPNAME} ${USER_INFO[0]}
done

