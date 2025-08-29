import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";
import { MemoryGame } from "./memory-game";
import { PuzzleGame } from "./puzzle-game";

export default function GamesPage() {
  return (
    <div className="w-full">
      <Tabs defaultValue="memory" className="w-full">
        <TabsList className="grid w-full grid-cols-2 max-w-md mx-auto">
          <TabsTrigger value="memory">Memory Game</TabsTrigger>
          <TabsTrigger value="puzzle">Word Scramble</TabsTrigger>
        </TabsList>
        <TabsContent value="memory">
          <div className="mt-6 flex flex-col items-center">
            <h2 className="text-2xl font-headline font-semibold mb-2">Match the Pairs!</h2>
            <p className="text-muted-foreground mb-6">Click the cards to find matching icons.</p>
            <MemoryGame />
          </div>
        </TabsContent>
        <TabsContent value="puzzle">
           <div className="mt-6 flex flex-col items-center">
            <h2 className="text-2xl font-headline font-semibold mb-2">Unscramble the Word!</h2>
            <p className="text-muted-foreground mb-6">Can you figure out the education-themed word?</p>
            <PuzzleGame />
          </div>
        </TabsContent>
      </Tabs>
    </div>
  );
}
