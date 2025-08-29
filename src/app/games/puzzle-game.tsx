"use client";

import { useState, useMemo, useEffect } from "react";
import { Input } from "@/components/ui/input";
import { Button } from "@/components/ui/button";
import { Card, CardContent } from "@/components/ui/card";
import { AlertCircle, CheckCircle } from "lucide-react";

const words = ["EDUCATION", "KNOWLEDGE", "LEARNING", "ASSESSMENT", "PERFORMANCE", "FEEDBACK", "PROJECT"];

const scrambleWord = (word: string) => {
  const a = word.split("");
  const n = a.length;
  for (let i = n - 1; i > 0; i--) {
    const j = Math.floor(Math.random() * (i + 1));
    [a[i], a[j]] = [a[j], a[i]];
  }
  return a.join("");
};

export function PuzzleGame() {
  const [currentWord, setCurrentWord] = useState("");
  const [scrambledWord, setScrambledWord] = useState("");
  const [guess, setGuess] = useState("");
  const [message, setMessage] = useState({ text: "", type: "" });

  const generateNewPuzzle = () => {
    const newWord = words[Math.floor(Math.random() * words.length)];
    setCurrentWord(newWord);
    let newScrambled = scrambleWord(newWord);
    while (newScrambled === newWord) {
        newScrambled = scrambleWord(newWord);
    }
    setScrambledWord(newScrambled);
    setGuess("");
    setMessage({ text: "", type: "" });
  };
  
  useEffect(() => {
    generateNewPuzzle();
  }, []);

  const handleGuess = () => {
    if (guess.toUpperCase() === currentWord) {
      setMessage({ text: "Correct! Well done!", type: "success" });
    } else {
      setMessage({ text: "Not quite, try again!", type: "error" });
    }
  };

  const handleKeyPress = (e: React.KeyboardEvent<HTMLInputElement>) => {
    if (e.key === 'Enter') {
        handleGuess();
    }
  }

  return (
    <Card className="w-full max-w-md p-6 md:p-8 text-center shadow-lg">
      <CardContent className="p-0">
        <p className="text-4xl md:text-5xl font-bold tracking-widest text-primary mb-6">
          {scrambledWord}
        </p>
        <div className="flex w-full items-center space-x-2">
          <Input
            type="text"
            placeholder="Your guess..."
            value={guess}
            onChange={(e) => setGuess(e.target.value)}
            onKeyPress={handleKeyPress}
            className="text-lg text-center"
          />
          <Button onClick={handleGuess}>Guess</Button>
        </div>
        {message.text && (
            <div className={`mt-4 flex items-center justify-center gap-2 p-3 rounded-md text-sm ${message.type === 'success' ? 'bg-green-100 text-green-800' : 'bg-red-100 text-red-800'}`}>
                {message.type === 'success' ? <CheckCircle className="h-5 w-5"/> : <AlertCircle className="h-5 w-5"/>}
                {message.text}
            </div>
        )}
        <Button variant="link" onClick={generateNewPuzzle} className="mt-4">
          New Word
        </Button>
      </CardContent>
    </Card>
  );
}
