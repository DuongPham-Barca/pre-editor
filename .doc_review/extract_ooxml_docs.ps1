param(
    [Parameter(Mandatory = $true)] [string[]]$InputPaths,
    [Parameter(Mandatory = $true)] [string]$OutputDirectory
)

$ErrorActionPreference = 'Stop'
trap {
    Write-Host $_.InvocationInfo.PositionMessage
    Write-Host $_.Exception.Message
    break
}
Add-Type -AssemblyName System.IO.Compression.FileSystem
New-Item -ItemType Directory -Force -Path $OutputDirectory | Out-Null

function Read-ZipXml($Zip, [string]$EntryName) {
    $entry = $Zip.GetEntry($EntryName)
    if ($null -eq $entry) { return $null }
    $stream = $entry.Open()
    $reader = [System.IO.StreamReader]::new($stream, [System.Text.Encoding]::UTF8, $true)
    try {
        $xml = [xml]$reader.ReadToEnd()
        return ,$xml
    }
    finally {
        $reader.Dispose()
        $stream.Dispose()
    }
}

function New-WordNs($Xml) {
    $ns = [System.Xml.XmlNamespaceManager]::new($Xml.NameTable)
    $ns.AddNamespace('w', 'http://schemas.openxmlformats.org/wordprocessingml/2006/main')
    $ns.AddNamespace('r', 'http://schemas.openxmlformats.org/officeDocument/2006/relationships')
    return ,$ns
}

function Get-NodeText($Node, $Ns) {
    $parts = [System.Collections.Generic.List[string]]::new()
    foreach ($item in $Node.SelectNodes('.//w:t | .//w:delText | .//w:tab | .//w:br | .//w:cr | .//w:noBreakHyphen', $Ns)) {
        switch ($item.LocalName) {
            't' { $parts.Add($item.InnerText) }
            'delText' { $parts.Add("[-$($item.InnerText)-]") }
            'tab' { $parts.Add("`t") }
            'br' { $parts.Add("`n") }
            'cr' { $parts.Add("`n") }
            'noBreakHyphen' { $parts.Add('-') }
        }
    }
    return (($parts -join '') -replace "[\r\n]+$", '').Trim()
}

function Get-StyleMap($StylesXml) {
    $map = @{}
    if ($null -eq $StylesXml) { return $map }
    $ns = New-WordNs $StylesXml
    foreach ($style in $StylesXml.SelectNodes('//w:style', $ns)) {
        $id = $style.GetAttribute('styleId', 'http://schemas.openxmlformats.org/wordprocessingml/2006/main')
        $nameNode = $style.SelectSingleNode('./w:name', $ns)
        if ($nameNode) {
            $map[$id] = $nameNode.GetAttribute('val', 'http://schemas.openxmlformats.org/wordprocessingml/2006/main')
        }
    }
    return $map
}

function Get-ParagraphInfo($Paragraph, $Ns, $StyleMap, [int]$Index) {
    $styleId = ''
    $styleNode = $Paragraph.SelectSingleNode('./w:pPr/w:pStyle', $Ns)
    if ($styleNode) { $styleId = $styleNode.GetAttribute('val', 'http://schemas.openxmlformats.org/wordprocessingml/2006/main') }
    $styleName = if ($StyleMap.ContainsKey($styleId)) { $StyleMap[$styleId] } else { $styleId }
    $numId = ''
    $level = ''
    $numNode = $Paragraph.SelectSingleNode('./w:pPr/w:numPr/w:numId', $Ns)
    $levelNode = $Paragraph.SelectSingleNode('./w:pPr/w:numPr/w:ilvl', $Ns)
    if ($numNode) { $numId = $numNode.GetAttribute('val', 'http://schemas.openxmlformats.org/wordprocessingml/2006/main') }
    if ($levelNode) { $level = $levelNode.GetAttribute('val', 'http://schemas.openxmlformats.org/wordprocessingml/2006/main') }
    return [PSCustomObject]@{
        index = $Index
        style_id = $styleId
        style = $styleName
        numbered = [bool]$numNode
        numbering_id = $numId
        numbering_level = $level
        text = Get-NodeText $Paragraph $Ns
    }
}

function Process-Container($Container, $Ns, $StyleMap, [ref]$ParagraphIndex, [ref]$TableIndex, $Paragraphs, $Tables, $Ordered) {
    foreach ($child in $Container.ChildNodes) {
        if ($child.NamespaceURI -ne 'http://schemas.openxmlformats.org/wordprocessingml/2006/main') { continue }
        switch ($child.LocalName) {
            'p' {
                $ParagraphIndex.Value++
                $p = Get-ParagraphInfo $child $Ns $StyleMap $ParagraphIndex.Value
                if (-not [string]::IsNullOrWhiteSpace($p.text)) {
                    $Paragraphs.Add($p)
                    $Ordered.Add([PSCustomObject]@{ type = 'paragraph'; value = $p })
                }
            }
            'tbl' {
                $TableIndex.Value++
                $rows = [System.Collections.Generic.List[object]]::new()
                foreach ($rowNode in $child.SelectNodes('./w:tr', $Ns)) {
                    $cells = [System.Collections.Generic.List[string]]::new()
                    foreach ($cellNode in $rowNode.SelectNodes('./w:tc', $Ns)) {
                        $cellParagraphs = [System.Collections.Generic.List[string]]::new()
                        foreach ($cp in $cellNode.SelectNodes('.//w:p', $Ns)) {
                            $ct = Get-NodeText $cp $Ns
                            if (-not [string]::IsNullOrWhiteSpace($ct)) { $cellParagraphs.Add($ct) }
                        }
                        $cells.Add(($cellParagraphs -join ' / '))
                    }
                    $rows.Add($cells.ToArray())
                }
                $table = [PSCustomObject]@{ index = $TableIndex.Value; rows = $rows.Count; data = $rows.ToArray() }
                $Tables.Add($table)
                $Ordered.Add([PSCustomObject]@{ type = 'table'; value = $table })
            }
            'sdt' {
                $content = $child.SelectSingleNode('./w:sdtContent', $Ns)
                if ($content) { Process-Container $content $Ns $StyleMap $ParagraphIndex $TableIndex $Paragraphs $Tables $Ordered }
            }
        }
    }
}

$documentResults = [System.Collections.Generic.List[object]]::new()
$fileIndex = 0
foreach ($path in $InputPaths) {
    $fileIndex++
    $zip = [System.IO.Compression.ZipFile]::OpenRead($path)
    try {
        $docXml = Read-ZipXml $zip 'word/document.xml'
        if ($null -eq $docXml) { throw "Missing word/document.xml in $path" }
        $styles = Get-StyleMap (Read-ZipXml $zip 'word/styles.xml')
        $ns = New-WordNs $docXml
        $body = $docXml.SelectSingleNode('//w:body', $ns)

        $paragraphs = [System.Collections.Generic.List[object]]::new()
        $tables = [System.Collections.Generic.List[object]]::new()
        $ordered = [System.Collections.Generic.List[object]]::new()
        [int]$pi = 0
        [int]$ti = 0
        Process-Container $body $ns $styles ([ref]$pi) ([ref]$ti) $paragraphs $tables $ordered

        $comments = [System.Collections.Generic.List[object]]::new()
        $commentsXml = Read-ZipXml $zip 'word/comments.xml'
        if ($commentsXml) {
            $cns = New-WordNs $commentsXml
            foreach ($c in $commentsXml.SelectNodes('//w:comment', $cns)) {
                $comments.Add([PSCustomObject]@{
                    id = $c.GetAttribute('id', 'http://schemas.openxmlformats.org/wordprocessingml/2006/main')
                    author = $c.GetAttribute('author', 'http://schemas.openxmlformats.org/wordprocessingml/2006/main')
                    date = $c.GetAttribute('date', 'http://schemas.openxmlformats.org/wordprocessingml/2006/main')
                    text = Get-NodeText $c $cns
                })
            }
        }

        $headersFooters = [System.Collections.Generic.List[object]]::new()
        foreach ($entry in $zip.Entries | Where-Object { $_.FullName -match '^word/(header|footer)\d+\.xml$' }) {
            $hfXml = Read-ZipXml $zip $entry.FullName
            $hfNs = New-WordNs $hfXml
            $headersFooters.Add([PSCustomObject]@{
                part = $entry.FullName
                text = (($hfXml.SelectNodes('//w:p', $hfNs) | ForEach-Object { Get-NodeText $_ $hfNs } | Where-Object { $_ }) -join ' | ')
            })
        }

        $appXml = Read-ZipXml $zip 'docProps/app.xml'
        $pages = $null; $words = $null; $characters = $null
        if ($appXml) {
            $pagesNode = $appXml.SelectSingleNode("//*[local-name()='Pages']")
            $wordsNode = $appXml.SelectSingleNode("//*[local-name()='Words']")
            $charsNode = $appXml.SelectSingleNode("//*[local-name()='Characters']")
            if ($pagesNode) { $pages = [int]$pagesNode.InnerText }
            if ($wordsNode) { $words = [int]$wordsNode.InnerText }
            if ($charsNode) { $characters = [int]$charsNode.InnerText }
        }

        $result = [PSCustomObject]@{
            source = $path
            title = [System.IO.Path]::GetFileNameWithoutExtension($path)
            pages = $pages
            words = $words
            characters = $characters
            paragraphs_count = $paragraphs.Count
            tables_count = $tables.Count
            images_count = ($zip.Entries | Where-Object { $_.FullName -match '^word/media/' }).Count
            comments_count = $comments.Count
            tracked_insertions_count = $docXml.SelectNodes('//w:ins', $ns).Count
            tracked_deletions_count = $docXml.SelectNodes('//w:del', $ns).Count
            hyperlinks_count = $docXml.SelectNodes('//w:hyperlink', $ns).Count
            footnotes_present = [bool]$zip.GetEntry('word/footnotes.xml')
            endnotes_present = [bool]$zip.GetEntry('word/endnotes.xml')
            headers_footers = $headersFooters.ToArray()
            paragraphs = $paragraphs.ToArray()
            tables = $tables.ToArray()
            comments = $comments.ToArray()
        }
        $documentResults.Add($result)

        $stem = ('{0:D2}_{1}' -f $fileIndex, ($result.title -replace '[^\p{L}\p{Nd}._-]+', '_'))
        $jsonPath = Join-Path $OutputDirectory ($stem + '.json')
        $textPath = Join-Path $OutputDirectory ($stem + '.txt')
        $result | ConvertTo-Json -Depth 15 | Set-Content -LiteralPath $jsonPath -Encoding UTF8

        $lines = [System.Collections.Generic.List[string]]::new()
        $lines.Add("TITLE: $($result.title)")
        $lines.Add("PAGES: $($result.pages); WORDS: $($result.words); TABLES: $($result.tables_count); IMAGES: $($result.images_count); COMMENTS: $($result.comments_count); INSERTIONS: $($result.tracked_insertions_count); DELETIONS: $($result.tracked_deletions_count)")
        $lines.Add('')
        foreach ($item in $ordered) {
            if ($item.type -eq 'paragraph') {
                $p = $item.value
                $tag = if ($p.style) { $p.style } else { 'paragraph' }
                if ($p.numbered) { $tag += ", list L$($p.numbering_level)" }
                $lines.Add("[$tag] $($p.text)")
            }
            else {
                $t = $item.value
                $lines.Add("[TABLE $($t.index), $($t.rows) rows]")
                foreach ($row in $t.data) { $lines.Add(($row -join ' | ')) }
                $lines.Add('[/TABLE]')
            }
        }
        if ($headersFooters.Count -gt 0) {
            $lines.Add(''); $lines.Add('=== HEADERS / FOOTERS ===')
            foreach ($hf in $headersFooters) { $lines.Add("[$($hf.part)] $($hf.text)") }
        }
        if ($comments.Count -gt 0) {
            $lines.Add(''); $lines.Add('=== COMMENTS ===')
            foreach ($c in $comments) { $lines.Add("[$($c.author)] $($c.text)") }
        }
        $lines | Set-Content -LiteralPath $textPath -Encoding UTF8
    }
    finally { $zip.Dispose() }
}

$documentResults | Select-Object title,pages,words,paragraphs_count,tables_count,images_count,comments_count,tracked_insertions_count,tracked_deletions_count,hyperlinks_count | Format-Table -AutoSize
