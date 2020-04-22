<?php

if(isset($_POST['submit'])) {
    $name = $_POST['fname'];
    $mailFrom = $_POST['mail'];
    $mailTo = $_POST['recip'];
    $object = $_POST['obj'];
    $message = $_POST['subject'];

    $headers = "From: ".$mailFrom;
    $txt = "You have received an e-mail from ".$name.".\n\n".$message




    mail($mailTo, $object, $txt, $headers);
    header("Location: index.php?mailsend");
    
}